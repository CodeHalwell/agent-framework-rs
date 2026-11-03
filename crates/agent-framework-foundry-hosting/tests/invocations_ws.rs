//! The `/invocations_ws` WebSocket route (feature `websocket`).
#![cfg(feature = "websocket")]

use std::time::Duration;

use agent_framework_foundry_hosting::invocations_ws::{InvocationsWs, Message, WsSession};
use futures::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::{self, client::IntoClientRequest};

/// Echoes text with the caller's call id; `boom` fails, `bye` ends cleanly.
async fn echo(mut session: WsSession) -> agent_framework_core::error::Result<()> {
    let call = session.context().call_id.clone().unwrap_or_default();
    while let Some(message) = session.recv().await {
        let Message::Text(text) = message else {
            continue;
        };
        match text.as_str() {
            "boom" => return Err(agent_framework_core::error::Error::Other("boom".into())),
            "bye" => {
                session.send(Message::Text("goodbye".into())).await?;
                return Ok(());
            }
            other => {
                session
                    .send(Message::Text(format!("{call}:{other}").into()))
                    .await?
            }
        }
    }
    Ok(())
}

async fn serve(ping: Option<Duration>) -> String {
    let mut ws = InvocationsWs::new(echo);
    if let Some(ping) = ping {
        ws = ws.ping_interval(ping);
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, ws.into_router()).await.unwrap() });
    format!("ws://{addr}/invocations_ws")
}

type Client =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

async fn connect(url: &str) -> Client {
    let mut request = url.into_client_request().unwrap();
    request
        .headers_mut()
        .insert("x-agent-foundry-call-id", "call-7".parse().unwrap());
    tokio_tungstenite::connect_async(request).await.unwrap().0
}

/// The next data or close frame (pings are answered by the client).
async fn next(client: &mut Client) -> tungstenite::Message {
    loop {
        let message = tokio::time::timeout(Duration::from_secs(5), client.next())
            .await
            .expect("a frame")
            .expect("open")
            .unwrap();
        if !matches!(
            message,
            tungstenite::Message::Ping(_) | tungstenite::Message::Pong(_)
        ) {
            return message;
        }
    }
}

fn close_code(message: tungstenite::Message) -> u16 {
    match message {
        tungstenite::Message::Close(Some(frame)) => frame.code.into(),
        other => panic!("expected a close frame, got {other:?}"),
    }
}

#[tokio::test]
async fn messages_reach_the_handler_with_the_request_context() {
    let url = serve(None).await;
    let mut client = connect(&url).await;
    client.send(tungstenite::Message::text("hi")).await.unwrap();
    assert_eq!(
        next(&mut client).await,
        tungstenite::Message::text("call-7:hi")
    );
    client
        .send(tungstenite::Message::text("bye"))
        .await
        .unwrap();
    assert_eq!(
        next(&mut client).await,
        tungstenite::Message::text("goodbye")
    );
    assert_eq!(
        close_code(next(&mut client).await),
        1000,
        "a clean return closes normally"
    );
}

#[tokio::test]
async fn a_failing_handler_closes_with_an_internal_error() {
    let url = serve(None).await;
    let mut client = connect(&url).await;
    client
        .send(tungstenite::Message::text("boom"))
        .await
        .unwrap();
    let message = next(&mut client).await;
    let tungstenite::Message::Close(Some(frame)) = &message else {
        panic!("{message:?}");
    };
    assert_eq!(u16::from(frame.code), 1011);
    assert_eq!(frame.reason.as_str(), "Internal server error");
}

#[tokio::test]
async fn keepalive_pings_are_sent_when_configured() {
    let url = serve(Some(Duration::from_millis(50))).await;
    let mut client = connect(&url).await;
    let ping = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if let Some(Ok(tungstenite::Message::Ping(_))) = client.next().await {
                return;
            }
        }
    })
    .await;
    assert!(ping.is_ok(), "a ping within the interval");
    // A client closing ends the handler (its receive yields None).
    client.close(None).await.unwrap();
}

#[test]
fn the_ping_interval_comes_from_the_environment() {
    assert_eq!(InvocationsWs::ping_interval_from(|_| None).unwrap(), None);
    assert_eq!(
        InvocationsWs::ping_interval_from(|_| Some("0".into())).unwrap(),
        None
    );
    assert_eq!(
        InvocationsWs::ping_interval_from(|_| Some("2.5".into())).unwrap(),
        Some(Duration::from_millis(2500))
    );
    assert!(InvocationsWs::ping_interval_from(|_| Some("-1".into())).is_err());
    assert!(InvocationsWs::ping_interval_from(|_| Some("x".into())).is_err());
}

/// Sends many frames without ever reading.
async fn chatty(session: WsSession) -> agent_framework_core::error::Result<()> {
    for i in 0..500 {
        session
            .send(Message::Text(format!("out {i}").into()))
            .await?;
    }
    Ok(())
}

#[tokio::test]
async fn a_handler_that_does_not_read_cannot_stall_its_output() {
    let ws = InvocationsWs::new(chatty).ping_interval(Duration::from_millis(20));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, ws.into_router()).await.unwrap() });
    let client = connect(&format!("ws://{addr}/invocations_ws")).await;
    let (mut sink, mut stream) = client.split();
    // The client floods input the handler never reads...
    let flood = tokio::spawn(async move {
        for i in 0..2000 {
            if sink
                .send(tungstenite::Message::text(format!("in {i}")))
                .await
                .is_err()
            {
                break;
            }
        }
    });
    // ...while reading everything the handler sends, then the close.
    let received = tokio::time::timeout(Duration::from_secs(10), async {
        let mut texts = 0;
        while let Some(Ok(message)) = stream.next().await {
            match message {
                tungstenite::Message::Text(_) => texts += 1,
                tungstenite::Message::Close(frame) => {
                    return (texts, frame.map(|f| u16::from(f.code)))
                }
                _ => {}
            }
        }
        (texts, None)
    })
    .await
    .expect("no deadlock: the output drains while input is backed up");
    assert_eq!(received, (500, Some(1000)));
    flood.abort();
}

#[tokio::test]
async fn a_client_disconnect_stops_an_output_only_handler() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    let stopped = Arc::new(AtomicBool::new(false));
    let flag = stopped.clone();
    // Streams forever without reading, until a send fails.
    let ws = InvocationsWs::new(move |session: WsSession| {
        let flag = flag.clone();
        async move {
            let mut i = 0u64;
            while session
                .send(Message::Text(format!("tick {i}").into()))
                .await
                .is_ok()
            {
                i += 1;
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            flag.store(true, Ordering::SeqCst);
            Ok(())
        }
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, ws.into_router()).await.unwrap() });
    let mut client = connect(&format!("ws://{addr}/invocations_ws")).await;
    assert!(matches!(
        next(&mut client).await,
        tungstenite::Message::Text(_)
    ));
    client.close(None).await.unwrap();
    drop(client);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while !stopped.load(Ordering::SeqCst) {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the handler's sends must fail once the client is gone"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}
