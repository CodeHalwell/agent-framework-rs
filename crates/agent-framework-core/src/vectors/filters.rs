//! Portable vector-store filter expressions.
//!
//! Ports upstream's `agent_framework._vector_filters` (#8115). A filter is a
//! tree of [`Filter`] leaves combined by [`FilterGroup`] nodes, built once by
//! the caller and translated by each connector into its own dialect — OData
//! for Azure AI Search, a JSON filter for a document store, and so on. The
//! same expression therefore survives a change of provider, which a
//! hand-written dialect string does not.
//!
//! # Operator semantics
//!
//! * [`FilterOperator::Eq`] / [`FilterOperator::Ne`] compare one field value.
//!   A boolean is never equal to a number, and two numbers compare by
//!   numeric value (so `1` matches `1.0`).
//! * [`FilterOperator::Gt`] / `Gte` / `Lt` / `Lte` are ordered scalar
//!   comparisons: numbers compare numerically and strings lexicographically.
//!   Anything else is a type error rather than an arbitrary ordering — which
//!   also means an ISO-8601 timestamp compares correctly *because* that
//!   encoding sorts lexicographically, and a non-ISO date format does not.
//! * [`FilterOperator::Between`] is inclusive and takes exactly
//!   `[lower, upper]`.
//! * [`FilterOperator::In`] / `NotIn` test whether the field value occurs in
//!   the supplied array.
//! * [`FilterOperator::Contains`] tests whether an array field contains one
//!   supplied value; `ContainsAny` / `ContainsAll` test it against an array
//!   of values.
//! * [`FilterOperator::IsNull`] / `IsNotNull` require the field to be
//!   present; use [`FilterOperator::Exists`] to test presence alone.
//! * [`FilterOperator::StartsWith`] / `EndsWith` / `ContainsText` require
//!   string operands.
//!
//! **A missing field yields `false` for every operator except `Exists`** —
//! including `Ne`, which is why negating a filter and testing inequality are
//! not the same thing. Use a [`FilterGroup`] with
//! [`FilterGroupOperator::Not`] for explicit negation.
//!
//! A provider-specific operator is namespaced
//! ([`FilterOperator::Provider`], e.g. `azure_ai_search.match`) and is
//! interpreted only by the connector that defines it; every other connector
//! rejects it rather than guessing.
//!
//! # Divergence: no `Param`
//!
//! Upstream's module also carries a `Param` type: a late-bound placeholder
//! that lets a filter be declared once and have its values supplied later by
//! an agent tool call, complete with a generated JSON Schema for the tool's
//! arguments. That machinery exists because the filter is declared in Python
//! *before* the values are known and there is no type to carry them. A Rust
//! caller building a filter in the tool body already has the arguments in
//! typed form and constructs the expression with them, so the placeholder,
//! its schema derivation, and the omission rules that follow from it have no
//! landing site here. Everything else in upstream's module — the operator
//! set, the shape validation, and the structural limits — is ported.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};

/// Maximum nesting depth of a filter expression.
///
/// Upstream's `_MAX_FILTER_DEPTH`. A tree deeper than this is refused at
/// construction, so no connector has to defend its translator against
/// unbounded recursion on a caller-supplied expression.
pub const MAX_FILTER_DEPTH: usize = 8;

/// Maximum number of nodes (leaves plus groups) in a filter expression.
///
/// Upstream's `_MAX_FILTER_NODES`.
pub const MAX_FILTER_NODES: usize = 64;

/// How one field is tested.
///
/// The standard operators are the portable set every connector is expected to
/// translate or explicitly refuse. [`FilterOperator::Provider`] carries a
/// namespaced operator that only its own connector understands.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum FilterOperator {
    /// Field equals the value.
    Eq,
    /// Field is present and does not equal the value.
    Ne,
    /// Field is greater than the value.
    Gt,
    /// Field is greater than or equal to the value.
    Gte,
    /// Field is less than the value.
    Lt,
    /// Field is less than or equal to the value.
    Lte,
    /// Field lies within `[lower, upper]`, inclusive.
    Between,
    /// Field occurs in the supplied array.
    In,
    /// Field does not occur in the supplied array.
    NotIn,
    /// Field is present and null.
    IsNull,
    /// Field is present and not null.
    IsNotNull,
    /// Field is present, whatever its value.
    Exists,
    /// Array field contains the supplied value.
    Contains,
    /// Array field contains at least one of the supplied values.
    ContainsAny,
    /// Array field contains all of the supplied values.
    ContainsAll,
    /// String field starts with the supplied string.
    StartsWith,
    /// String field ends with the supplied string.
    EndsWith,
    /// String field contains the supplied substring.
    ContainsText,
    /// A connector-specific operator, namespaced as `connector.operator`.
    ///
    /// Constructed through [`FilterOperator::provider`], which enforces the
    /// namespacing rule, so a typo in a standard operator name cannot
    /// silently become a provider operator that every connector then rejects
    /// at search time.
    Provider(String),
}

impl FilterOperator {
    /// The operator's wire name.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Gt => "gt",
            Self::Gte => "gte",
            Self::Lt => "lt",
            Self::Lte => "lte",
            Self::Between => "between",
            Self::In => "in",
            Self::NotIn => "not_in",
            Self::IsNull => "is_null",
            Self::IsNotNull => "is_not_null",
            Self::Exists => "exists",
            Self::Contains => "contains",
            Self::ContainsAny => "contains_any",
            Self::ContainsAll => "contains_all",
            Self::StartsWith => "starts_with",
            Self::EndsWith => "ends_with",
            Self::ContainsText => "contains_text",
            Self::Provider(name) => name,
        }
    }

    /// Build a provider-specific operator.
    ///
    /// The name must be namespaced — two or more `[a-z][a-z0-9_]*` segments
    /// joined by `.` — matching upstream's `_PROVIDER_OPERATOR_PATTERN`.
    pub fn provider(name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        if !is_namespaced_operator(&name) {
            return Err(Error::Configuration(format!(
                "provider filter operator '{name}' must be namespaced as \
                 `connector.operator` (lowercase segments joined by '.')"
            )));
        }
        Ok(Self::Provider(name))
    }

    /// Whether this is one of the portable operators (as opposed to a
    /// provider-specific one).
    pub fn is_standard(&self) -> bool {
        !matches!(self, Self::Provider(_))
    }

    fn takes_no_value(&self) -> bool {
        matches!(self, Self::IsNull | Self::IsNotNull | Self::Exists)
    }

    fn takes_text_value(&self) -> bool {
        matches!(self, Self::StartsWith | Self::EndsWith | Self::ContainsText)
    }

    fn takes_array_value(&self) -> bool {
        matches!(
            self,
            Self::Between | Self::In | Self::NotIn | Self::ContainsAny | Self::ContainsAll
        )
    }
}

impl std::str::FromStr for FilterOperator {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        Ok(match value {
            "eq" => Self::Eq,
            "ne" => Self::Ne,
            "gt" => Self::Gt,
            "gte" => Self::Gte,
            "lt" => Self::Lt,
            "lte" => Self::Lte,
            "between" => Self::Between,
            "in" => Self::In,
            "not_in" => Self::NotIn,
            "is_null" => Self::IsNull,
            "is_not_null" => Self::IsNotNull,
            "exists" => Self::Exists,
            "contains" => Self::Contains,
            "contains_any" => Self::ContainsAny,
            "contains_all" => Self::ContainsAll,
            "starts_with" => Self::StartsWith,
            "ends_with" => Self::EndsWith,
            "contains_text" => Self::ContainsText,
            other => Self::provider(other)?,
        })
    }
}

impl TryFrom<String> for FilterOperator {
    type Error = Error;

    fn try_from(value: String) -> Result<Self> {
        value.parse()
    }
}

impl From<FilterOperator> for String {
    fn from(value: FilterOperator) -> Self {
        value.as_str().to_string()
    }
}

impl std::fmt::Display for FilterOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Whether `name` is a valid namespaced provider operator: two or more
/// `[a-z][a-z0-9_]*` segments joined by `.`.
fn is_namespaced_operator(name: &str) -> bool {
    let mut segments = 0usize;
    for segment in name.split('.') {
        let mut chars = segment.chars();
        match chars.next() {
            Some(c) if c.is_ascii_lowercase() => {}
            _ => return false,
        }
        if !chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
            return false;
        }
        segments += 1;
    }
    segments >= 2
}

/// How a [`FilterGroup`]'s children combine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FilterGroupOperator {
    /// Every child must match.
    And,
    /// At least one child must match.
    Or,
    /// The single child must not match.
    Not,
}

impl FilterGroupOperator {
    /// The operator's wire name.
    pub fn as_str(&self) -> &str {
        match self {
            Self::And => "and",
            Self::Or => "or",
            Self::Not => "not",
        }
    }
}

/// One field test.
///
/// Built through [`Filter::new`] or one of the per-operator constructors,
/// both of which validate the field name and the value's shape, so a
/// connector can translate a `Filter` without re-checking that `Between`
/// carries two bounds or that `StartsWith` carries a string.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Filter {
    /// The logical field name, optionally a `.`-separated path into it where
    /// the provider supports one.
    pub field_name: String,
    /// How the field is tested.
    pub operator: FilterOperator,
    /// The operand, or `None` for the operators that take none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
}

impl Filter {
    /// Build a filter, validating the field name and the value's shape
    /// against the operator.
    pub fn new(
        field_name: impl Into<String>,
        operator: FilterOperator,
        value: Option<Value>,
    ) -> Result<Self> {
        let field_name = field_name.into();
        validate_field_name(&field_name)?;
        let filter = Self {
            field_name,
            operator,
            value,
        };
        filter.validate_value_shape()?;
        Ok(filter)
    }

    /// Field equals `value`.
    pub fn eq(field_name: impl Into<String>, value: impl Into<Value>) -> Result<Self> {
        Self::new(field_name, FilterOperator::Eq, Some(value.into()))
    }

    /// Field is present and differs from `value`.
    pub fn ne(field_name: impl Into<String>, value: impl Into<Value>) -> Result<Self> {
        Self::new(field_name, FilterOperator::Ne, Some(value.into()))
    }

    /// Field is greater than `value`.
    pub fn gt(field_name: impl Into<String>, value: impl Into<Value>) -> Result<Self> {
        Self::new(field_name, FilterOperator::Gt, Some(value.into()))
    }

    /// Field is greater than or equal to `value`.
    pub fn gte(field_name: impl Into<String>, value: impl Into<Value>) -> Result<Self> {
        Self::new(field_name, FilterOperator::Gte, Some(value.into()))
    }

    /// Field is less than `value`.
    pub fn lt(field_name: impl Into<String>, value: impl Into<Value>) -> Result<Self> {
        Self::new(field_name, FilterOperator::Lt, Some(value.into()))
    }

    /// Field is less than or equal to `value`.
    pub fn lte(field_name: impl Into<String>, value: impl Into<Value>) -> Result<Self> {
        Self::new(field_name, FilterOperator::Lte, Some(value.into()))
    }

    /// Field lies within `[lower, upper]`, inclusive.
    pub fn between(
        field_name: impl Into<String>,
        lower: impl Into<Value>,
        upper: impl Into<Value>,
    ) -> Result<Self> {
        Self::new(
            field_name,
            FilterOperator::Between,
            Some(Value::Array(vec![lower.into(), upper.into()])),
        )
    }

    /// Field occurs in `values`.
    pub fn any_of(
        field_name: impl Into<String>,
        values: impl IntoIterator<Item = Value>,
    ) -> Result<Self> {
        Self::new(
            field_name,
            FilterOperator::In,
            Some(Value::Array(values.into_iter().collect())),
        )
    }

    /// Field does not occur in `values`.
    pub fn none_of(
        field_name: impl Into<String>,
        values: impl IntoIterator<Item = Value>,
    ) -> Result<Self> {
        Self::new(
            field_name,
            FilterOperator::NotIn,
            Some(Value::Array(values.into_iter().collect())),
        )
    }

    /// Field is present, whatever its value.
    pub fn exists(field_name: impl Into<String>) -> Result<Self> {
        Self::new(field_name, FilterOperator::Exists, None)
    }

    /// Field is present and null.
    pub fn is_null(field_name: impl Into<String>) -> Result<Self> {
        Self::new(field_name, FilterOperator::IsNull, None)
    }

    /// Field is present and not null.
    pub fn is_not_null(field_name: impl Into<String>) -> Result<Self> {
        Self::new(field_name, FilterOperator::IsNotNull, None)
    }

    /// Array field contains `value`.
    pub fn contains(field_name: impl Into<String>, value: impl Into<Value>) -> Result<Self> {
        Self::new(field_name, FilterOperator::Contains, Some(value.into()))
    }

    /// Array field contains at least one of `values`.
    pub fn contains_any(
        field_name: impl Into<String>,
        values: impl IntoIterator<Item = Value>,
    ) -> Result<Self> {
        Self::new(
            field_name,
            FilterOperator::ContainsAny,
            Some(Value::Array(values.into_iter().collect())),
        )
    }

    /// Array field contains all of `values`.
    pub fn contains_all(
        field_name: impl Into<String>,
        values: impl IntoIterator<Item = Value>,
    ) -> Result<Self> {
        Self::new(
            field_name,
            FilterOperator::ContainsAll,
            Some(Value::Array(values.into_iter().collect())),
        )
    }

    /// String field starts with `value`.
    pub fn starts_with(field_name: impl Into<String>, value: impl Into<String>) -> Result<Self> {
        Self::new(
            field_name,
            FilterOperator::StartsWith,
            Some(Value::String(value.into())),
        )
    }

    /// String field ends with `value`.
    pub fn ends_with(field_name: impl Into<String>, value: impl Into<String>) -> Result<Self> {
        Self::new(
            field_name,
            FilterOperator::EndsWith,
            Some(Value::String(value.into())),
        )
    }

    /// String field contains the substring `value`.
    pub fn contains_text(field_name: impl Into<String>, value: impl Into<String>) -> Result<Self> {
        Self::new(
            field_name,
            FilterOperator::ContainsText,
            Some(Value::String(value.into())),
        )
    }

    /// Check the operand against what the operator accepts.
    ///
    /// Provider operators are not shape-checked here: their operand is
    /// defined by the connector, which is also the only thing that reads it.
    fn validate_value_shape(&self) -> Result<()> {
        if !self.operator.is_standard() {
            return Ok(());
        }
        let op = self.operator.as_str();
        if self.operator.takes_no_value() {
            return match self.value {
                None => Ok(()),
                Some(_) => Err(Error::Configuration(format!(
                    "filter operator '{op}' does not accept a value"
                ))),
            };
        }
        let Some(value) = self.value.as_ref() else {
            return Err(Error::Configuration(format!(
                "filter operator '{op}' requires a value"
            )));
        };
        if self.operator.takes_text_value() && !value.is_string() {
            return Err(Error::Configuration(format!(
                "filter operator '{op}' requires a string value"
            )));
        }
        if self.operator.takes_array_value() {
            let Some(items) = value.as_array() else {
                return Err(Error::Configuration(format!(
                    "filter operator '{op}' requires an array value"
                )));
            };
            if self.operator == FilterOperator::Between && items.len() != 2 {
                return Err(Error::Configuration(
                    "filter operator 'between' requires exactly two boundary values".into(),
                ));
            }
        }
        Ok(())
    }
}

/// A field name is one or more non-empty `.`-separated segments.
///
/// Upstream additionally requires every segment to be a Python identifier,
/// because there a field name addresses an attribute on a model class. Here a
/// record is a JSON object, whose keys are arbitrary strings, so the rule
/// would reject field names that are perfectly addressable — a storage name
/// like `content-type`, say. What the check has to catch is a name that
/// cannot be *split* unambiguously, which is an empty segment.
fn validate_field_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::Configuration(
            "filter field name cannot be empty".into(),
        ));
    }
    if name.split('.').any(str::is_empty) {
        return Err(Error::Configuration(format!(
            "filter field name '{name}' has an empty path segment"
        )));
    }
    Ok(())
}

/// Several filters combined by a boolean operator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FilterGroup {
    /// How the children combine.
    pub operator: FilterGroupOperator,
    /// The children. Non-empty; exactly one for
    /// [`FilterGroupOperator::Not`].
    pub filters: Vec<FilterExpression>,
}

impl FilterGroup {
    /// Build a group, rejecting an empty child list, a `Not` with anything
    /// but exactly one child, and a tree past the structural limits.
    pub fn new(operator: FilterGroupOperator, filters: Vec<FilterExpression>) -> Result<Self> {
        if filters.is_empty() {
            return Err(Error::Configuration(
                "a filter group requires at least one filter".into(),
            ));
        }
        if operator == FilterGroupOperator::Not && filters.len() != 1 {
            return Err(Error::Configuration(
                "a 'not' filter group requires exactly one filter".into(),
            ));
        }
        let group = FilterExpression::Group(Self { operator, filters });
        group.validate()?;
        match group {
            FilterExpression::Group(group) => Ok(group),
            FilterExpression::Condition(_) => unreachable!("built as a group"),
        }
    }

    /// [`Self::new`], wrapped as an expression ready to nest or search with.
    pub fn expression(
        operator: FilterGroupOperator,
        filters: Vec<FilterExpression>,
    ) -> Result<FilterExpression> {
        Self::new(operator, filters).map(FilterExpression::Group)
    }

    /// Every child must match.
    pub fn and(filters: Vec<FilterExpression>) -> Result<FilterExpression> {
        Self::expression(FilterGroupOperator::And, filters)
    }

    /// At least one child must match.
    pub fn or(filters: Vec<FilterExpression>) -> Result<FilterExpression> {
        Self::expression(FilterGroupOperator::Or, filters)
    }

    /// The child must not match.
    pub fn not(filter: FilterExpression) -> Result<FilterExpression> {
        Self::expression(FilterGroupOperator::Not, vec![filter])
    }
}

/// A filter tree: either one field test or a group of them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FilterExpression {
    /// One field test.
    Condition(Filter),
    /// Several tests combined.
    Group(FilterGroup),
}

impl From<Filter> for FilterExpression {
    fn from(value: Filter) -> Self {
        Self::Condition(value)
    }
}

impl FilterExpression {
    /// Enforce the structural limits ([`MAX_FILTER_DEPTH`],
    /// [`MAX_FILTER_NODES`]).
    ///
    /// Called by the constructors, so an expression built through them is
    /// already valid; call it directly on one that was deserialized, which
    /// bypasses them.
    pub fn validate(&self) -> Result<()> {
        let mut nodes = 0usize;
        self.validate_node(1, &mut nodes)
    }

    fn validate_node(&self, depth: usize, nodes: &mut usize) -> Result<()> {
        if depth > MAX_FILTER_DEPTH {
            return Err(Error::Configuration(format!(
                "filter expression is nested deeper than {MAX_FILTER_DEPTH} levels"
            )));
        }
        *nodes += 1;
        if *nodes > MAX_FILTER_NODES {
            return Err(Error::Configuration(format!(
                "filter expression has more than {MAX_FILTER_NODES} nodes"
            )));
        }
        match self {
            Self::Condition(filter) => {
                validate_field_name(&filter.field_name)?;
                filter.validate_value_shape()
            }
            Self::Group(group) => {
                if group.filters.is_empty() {
                    return Err(Error::Configuration(
                        "a filter group requires at least one filter".into(),
                    ));
                }
                if group.operator == FilterGroupOperator::Not && group.filters.len() != 1 {
                    return Err(Error::Configuration(
                        "a 'not' filter group requires exactly one filter".into(),
                    ));
                }
                for child in &group.filters {
                    child.validate_node(depth + 1, nodes)?;
                }
                Ok(())
            }
        }
    }

    /// Every leaf in this expression, in tree order.
    ///
    /// A connector uses this to reject an expression up front — an operator
    /// it cannot translate, or a field that is not filterable — instead of
    /// discovering it partway through building a query string.
    pub fn leaves(&self) -> Vec<&Filter> {
        let mut out = Vec::new();
        self.collect_leaves(&mut out);
        out
    }

    fn collect_leaves<'a>(&'a self, out: &mut Vec<&'a Filter>) {
        match self {
            Self::Condition(filter) => out.push(filter),
            Self::Group(group) => {
                for child in &group.filters {
                    child.collect_leaves(out);
                }
            }
        }
    }

    /// Evaluate this expression against `record`, a JSON object keyed by
    /// **storage** name.
    ///
    /// `resolve` maps a filter's logical field name onto the key to read from
    /// `record`, returning `None` for a field the collection does not
    /// declare — which is an error, not a non-match, since it is a mistake in
    /// the filter rather than a property of the data.
    ///
    /// This is the semantics upstream's `InMemoryCollection` implements, and
    /// the reference a connector's own translation is checked against.
    pub fn matches(
        &self,
        record: &Value,
        resolve: &dyn Fn(&str) -> Option<String>,
    ) -> Result<bool> {
        match self {
            Self::Group(group) => match group.operator {
                FilterGroupOperator::And => {
                    for child in &group.filters {
                        if !child.matches(record, resolve)? {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
                FilterGroupOperator::Or => {
                    for child in &group.filters {
                        if child.matches(record, resolve)? {
                            return Ok(true);
                        }
                    }
                    Ok(false)
                }
                // Indexing would panic, and this is reachable: `matches` is
                // public and `FilterExpression` is `Deserialize`, so a tree
                // that never went through a constructor (or `validate`) can
                // arrive here empty. An error is what the rest of this
                // method does with malformed input.
                FilterGroupOperator::Not => {
                    let inner = group.filters.first().ok_or_else(|| {
                        Error::Configuration(
                            "a 'not' filter group requires exactly one filter".into(),
                        )
                    })?;
                    Ok(!inner.matches(record, resolve)?)
                }
            },
            Self::Condition(filter) => evaluate_filter(filter, record, resolve),
        }
    }
}

fn evaluate_filter(
    filter: &Filter,
    record: &Value,
    resolve: &dyn Fn(&str) -> Option<String>,
) -> Result<bool> {
    if filter.field_name.contains('.') {
        return Err(Error::Configuration(format!(
            "nested filter field paths are not supported here: '{}'",
            filter.field_name
        )));
    }
    let storage_name = resolve(&filter.field_name).ok_or_else(|| {
        Error::Configuration(format!(
            "filter field '{}' is not part of the collection definition",
            filter.field_name
        ))
    })?;
    let actual = record.get(&storage_name);
    let op = &filter.operator;

    // Presence tests come first: they are the only operators that can see a
    // missing field as anything but a non-match.
    match op {
        FilterOperator::Exists => return Ok(actual.is_some()),
        FilterOperator::IsNull => return Ok(matches!(actual, Some(Value::Null))),
        FilterOperator::IsNotNull => {
            return Ok(actual.is_some_and(|v| !v.is_null()));
        }
        _ => {}
    }
    let Some(actual) = actual else {
        return Ok(false);
    };
    // A present null answers only equality; every ordered or textual
    // comparison against it is a non-match rather than a type error, matching
    // upstream.
    if actual.is_null() && !matches!(op, FilterOperator::Eq | FilterOperator::Ne) {
        return Ok(false);
    }
    let expected = filter.value.as_ref();
    let type_error = |what: &str| {
        Error::Configuration(format!(
            "filter operator '{}' cannot compare field '{}': {what}",
            op.as_str(),
            filter.field_name
        ))
    };

    Ok(match op {
        FilterOperator::Eq => values_equal(actual, expected.unwrap_or(&Value::Null)),
        FilterOperator::Ne => !values_equal(actual, expected.unwrap_or(&Value::Null)),
        FilterOperator::Gt | FilterOperator::Gte | FilterOperator::Lt | FilterOperator::Lte => {
            let expected = expected.ok_or_else(|| type_error("no value supplied"))?;
            let ordering =
                compare_values(actual, expected).ok_or_else(|| type_error("incomparable types"))?;
            match op {
                FilterOperator::Gt => ordering.is_gt(),
                FilterOperator::Gte => ordering.is_ge(),
                FilterOperator::Lt => ordering.is_lt(),
                _ => ordering.is_le(),
            }
        }
        FilterOperator::Between => {
            // Same reasoning as the `not` group above: an unvalidated
            // `between` can carry any number of bounds, and indexing them
            // would panic rather than report the malformed filter.
            let bounds = expected
                .and_then(Value::as_array)
                .filter(|bounds| bounds.len() == 2)
                .ok_or_else(|| type_error("'between' requires two bounds"))?;
            let lower = compare_values(actual, &bounds[0])
                .ok_or_else(|| type_error("incomparable types"))?;
            let upper = compare_values(actual, &bounds[1])
                .ok_or_else(|| type_error("incomparable types"))?;
            lower.is_ge() && upper.is_le()
        }
        FilterOperator::In | FilterOperator::NotIn => {
            let items = expected
                .and_then(Value::as_array)
                .ok_or_else(|| type_error("membership requires an array"))?;
            let found = items.iter().any(|item| values_equal(actual, item));
            if *op == FilterOperator::In {
                found
            } else {
                !found
            }
        }
        FilterOperator::Contains => {
            let items = actual
                .as_array()
                .ok_or_else(|| type_error("'contains' requires an array field"))?;
            let expected = expected.ok_or_else(|| type_error("no value supplied"))?;
            items.iter().any(|item| values_equal(item, expected))
        }
        FilterOperator::ContainsAny | FilterOperator::ContainsAll => {
            let items = actual
                .as_array()
                .ok_or_else(|| type_error("collection membership requires an array field"))?;
            let wanted = expected
                .and_then(Value::as_array)
                .ok_or_else(|| type_error("collection membership requires an array"))?;
            let mut hits = wanted
                .iter()
                .map(|want| items.iter().any(|item| values_equal(item, want)));
            if *op == FilterOperator::ContainsAny {
                hits.any(|hit| hit)
            } else {
                hits.all(|hit| hit)
            }
        }
        FilterOperator::StartsWith | FilterOperator::EndsWith | FilterOperator::ContainsText => {
            let haystack = actual
                .as_str()
                .ok_or_else(|| type_error("text comparison requires a string field"))?;
            let needle = expected
                .and_then(Value::as_str)
                .ok_or_else(|| type_error("text comparison requires a string value"))?;
            match op {
                FilterOperator::StartsWith => haystack.starts_with(needle),
                FilterOperator::EndsWith => haystack.ends_with(needle),
                _ => haystack.contains(needle),
            }
        }
        FilterOperator::Provider(name) => {
            return Err(Error::Configuration(format!(
                "filter operator '{name}' is provider-specific and is not understood here"
            )));
        }
        // Handled above, before the missing-field check.
        FilterOperator::Exists | FilterOperator::IsNull | FilterOperator::IsNotNull => {
            unreachable!("presence operators return above")
        }
    })
}

/// Equality with upstream's `filter_values_equal` semantics.
///
/// Two differences from `serde_json`'s own `==`, both deliberate:
///
/// * A boolean never equals a number. `serde_json` already keeps `Bool` and
///   `Number` in separate variants, so this falls out — but it is the rule
///   upstream had to write explicitly (Python's `True == 1`), and stating it
///   here keeps the two implementations comparable.
/// * Two numbers compare by **value**, so `1` equals `1.0`. `serde_json`
///   compares its internal representation, under which an integer and a float
///   of the same value are not equal — a caller filtering `eq: 1.0` against a
///   record that round-tripped through JSON as `1` would otherwise silently
///   match nothing.
fn values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => numbers_equal(a, b),
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| values_equal(x, y))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|other| values_equal(v, other)))
        }
        _ => left == right,
    }
}

/// Compare two JSON numbers by value.
///
/// Integers are compared **as integers**. Routing them through `f64` — the
/// obvious way to make `1` equal `1.0` — silently rounds past 2^53, so
/// `9007199254740992` and `9007199254740993` become the same number and a
/// filter on an id, a timestamp in nanoseconds, or any other large key
/// matches the wrong record. `f64` is used only when an operand is genuinely
/// non-integral, which is the case that needed the cross-type comparison in
/// the first place.
fn numbers_equal(a: &serde_json::Number, b: &serde_json::Number) -> bool {
    compare_numbers(a, b) == Some(std::cmp::Ordering::Equal)
}

/// The integer value of a JSON number, when it has one.
///
/// `i128` so that both `i64` and `u64` fit without a lossy step.
fn as_integer(n: &serde_json::Number) -> Option<i128> {
    n.as_i64()
        .map(i128::from)
        .or_else(|| n.as_u64().map(i128::from))
}

/// Order an integer against a float **exactly**.
///
/// Converting the integer to `f64` — the obvious way — rounds it past 2^53,
/// so `9007199254740993` would compare equal to the distinct float
/// `9007199254740992.0`. Comparing in the other direction instead is exact:
/// a finite float's integer part converts to `i128` without loss, and its
/// fraction then breaks the tie.
fn compare_integer_to_float(integer: i128, float: f64) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering;
    if float.is_nan() {
        return None;
    }
    // Outside `i128`'s range (infinities included) the float decides the
    // comparison on its own: no integer this function can receive reaches
    // that far.
    const I128_MAX_AS_F64: f64 = 170_141_183_460_469_231_731_687_303_715_884_105_728.0;
    if float >= I128_MAX_AS_F64 {
        return Some(Ordering::Less);
    }
    if float < -I128_MAX_AS_F64 {
        return Some(Ordering::Greater);
    }
    let truncated = float.trunc();
    Some(match integer.cmp(&(truncated as i128)) {
        // Same integer part, so the fraction decides: `3` is less than `3.5`
        // and greater than `3.0` only if that fraction is non-zero.
        Ordering::Equal => match (float - truncated).partial_cmp(&0.0)? {
            Ordering::Greater => Ordering::Less,
            Ordering::Less => Ordering::Greater,
            Ordering::Equal => Ordering::Equal,
        },
        other => other,
    })
}

/// Order two JSON numbers by value, with the same exact-integer rule as
/// [`numbers_equal`]: `gt`/`lt`/`between` on large integers must not be
/// decided by a rounded `f64`.
fn compare_numbers(a: &serde_json::Number, b: &serde_json::Number) -> Option<std::cmp::Ordering> {
    match (as_integer(a), as_integer(b)) {
        // Both integers, including a negative `i64` against a `u64` past
        // `i64::MAX`: `i128` holds both, so one comparison covers it.
        (Some(x), Some(y)) => Some(x.cmp(&y)),
        (Some(x), None) => compare_integer_to_float(x, b.as_f64()?),
        (None, Some(y)) => {
            compare_integer_to_float(y, a.as_f64()?).map(std::cmp::Ordering::reverse)
        }
        // Two floats, or a number no `f64` can hold (an arbitrary-precision
        // literal), which keeps its exact representational comparison rather
        // than being coerced into a lossy one.
        (None, None) => match (a.as_f64(), b.as_f64()) {
            (Some(x), Some(y)) => x.partial_cmp(&y),
            _ if a == b => Some(std::cmp::Ordering::Equal),
            _ => None,
        },
    }
}

/// Order two values, or `None` when they are not comparable.
///
/// Numbers compare numerically and strings lexicographically. Booleans,
/// nulls, arrays and objects have no portable ordering and are refused, so a
/// connector and this evaluator agree on what an ordered comparison means.
fn compare_values(left: &Value, right: &Value) -> Option<std::cmp::Ordering> {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => compare_numbers(a, b),
        (Value::String(a), Value::String(b)) => Some(a.as_str().cmp(b.as_str())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Resolve a logical name to itself — the identity mapping a collection
    /// without renamed fields provides.
    fn identity(name: &str) -> Option<String> {
        Some(name.to_string())
    }

    // region: construction and validation

    #[test]
    fn no_value_operators_reject_a_value() {
        assert!(Filter::exists("a").is_ok());
        let err = Filter::new("a", FilterOperator::Exists, Some(json!(1)))
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not accept a value"), "{err}");
    }

    #[test]
    fn value_operators_require_a_value() {
        let err = Filter::new("a", FilterOperator::Eq, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("requires a value"), "{err}");
    }

    #[test]
    fn between_requires_exactly_two_bounds() {
        assert!(Filter::between("a", 1, 5).is_ok());
        let err = Filter::new("a", FilterOperator::Between, Some(json!([1, 2, 3])))
            .unwrap_err()
            .to_string();
        assert!(err.contains("exactly two"), "{err}");
    }

    #[test]
    fn text_operators_require_a_string() {
        let err = Filter::new("a", FilterOperator::StartsWith, Some(json!(3)))
            .unwrap_err()
            .to_string();
        assert!(err.contains("requires a string value"), "{err}");
    }

    #[test]
    fn sequence_operators_require_an_array() {
        let err = Filter::new("a", FilterOperator::In, Some(json!("x")))
            .unwrap_err()
            .to_string();
        assert!(err.contains("requires an array value"), "{err}");
    }

    #[test]
    fn a_provider_operator_must_be_namespaced() {
        assert!(FilterOperator::provider("azure_ai_search.match").is_ok());
        // A misspelled standard operator is the case this catches: without
        // the namespacing rule `startswith` would parse as a provider
        // operator and fail only at search time.
        let err = FilterOperator::provider("startswith")
            .unwrap_err()
            .to_string();
        assert!(err.contains("must be namespaced"), "{err}");
        assert!(FilterOperator::provider("Azure.Match").is_err());
        assert!(FilterOperator::provider("azure..match").is_err());
    }

    #[test]
    fn an_unknown_operator_parses_only_when_namespaced() {
        assert_eq!(
            "azure_ai_search.match".parse::<FilterOperator>().unwrap(),
            FilterOperator::Provider("azure_ai_search.match".into())
        );
        assert!("nonsense".parse::<FilterOperator>().is_err());
        assert_eq!(
            "gte".parse::<FilterOperator>().unwrap(),
            FilterOperator::Gte
        );
    }

    #[test]
    fn field_names_reject_empty_segments() {
        assert!(Filter::exists("").is_err());
        assert!(Filter::exists("a..b").is_err());
        assert!(Filter::exists("a.b").is_ok());
        // A JSON key that is not a Python identifier is still addressable
        // here — the divergence documented on `validate_field_name`.
        assert!(Filter::exists("content-type").is_ok());
    }

    #[test]
    fn a_not_group_takes_exactly_one_child() {
        let a = Filter::eq("a", 1).unwrap().into();
        let b: FilterExpression = Filter::eq("b", 2).unwrap().into();
        assert!(FilterGroup::not(a).is_ok());
        assert!(FilterGroup::new(
            FilterGroupOperator::Not,
            vec![Filter::eq("a", 1).unwrap().into(), b]
        )
        .is_err());
    }

    #[test]
    fn an_empty_group_is_rejected() {
        assert!(FilterGroup::and(vec![]).is_err());
    }

    #[test]
    fn depth_and_node_limits_are_enforced() {
        let mut expr: FilterExpression = Filter::eq("a", 1).unwrap().into();
        for _ in 0..MAX_FILTER_DEPTH {
            match FilterGroup::not(expr.clone()) {
                Ok(next) => expr = next,
                Err(e) => {
                    assert!(e.to_string().contains("nested deeper"), "{e}");
                    return;
                }
            }
        }
        panic!("depth limit was never hit");
    }

    #[test]
    fn the_node_limit_counts_leaves_and_groups() {
        let leaves: Vec<FilterExpression> = (0..MAX_FILTER_NODES)
            .map(|i| Filter::eq(format!("f{i}"), i as i64).unwrap().into())
            .collect();
        let err = FilterGroup::and(leaves).unwrap_err().to_string();
        assert!(err.contains("more than"), "{err}");
    }

    #[test]
    fn a_deserialized_expression_can_be_revalidated() {
        // Deserialization bypasses the constructors, which is exactly why
        // `validate` is public.
        let raw = json!({"operator": "not", "filters": [
            {"field_name": "a", "operator": "eq", "value": 1},
            {"field_name": "b", "operator": "eq", "value": 2}
        ]});
        let expr: FilterExpression = serde_json::from_value(raw).unwrap();
        assert!(expr.validate().is_err());
    }

    #[test]
    fn round_trips_through_json() {
        let expr = FilterGroup::and(vec![
            Filter::eq("kind", "doc").unwrap().into(),
            FilterGroup::not(Filter::contains_text("body", "draft").unwrap().into()).unwrap(),
        ])
        .unwrap();
        let encoded = serde_json::to_value(&expr).unwrap();
        assert_eq!(encoded["operator"], "and");
        assert_eq!(encoded["filters"][0]["operator"], "eq");
        let decoded: FilterExpression = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, expr);
    }

    #[test]
    fn leaves_walks_the_whole_tree() {
        let expr = FilterGroup::or(vec![
            Filter::eq("a", 1).unwrap().into(),
            FilterGroup::and(vec![
                Filter::eq("b", 2).unwrap().into(),
                Filter::eq("c", 3).unwrap().into(),
            ])
            .unwrap(),
        ])
        .unwrap();
        let names: Vec<&str> = expr
            .leaves()
            .iter()
            .map(|f| f.field_name.as_str())
            .collect();
        assert_eq!(names, ["a", "b", "c"]);
    }

    // region: evaluation

    #[test]
    fn a_missing_field_matches_nothing_but_exists() {
        let record = json!({"present": 1});
        for filter in [
            Filter::eq("absent", 1).unwrap(),
            Filter::ne("absent", 1).unwrap(),
            Filter::gt("absent", 0).unwrap(),
            Filter::contains_text("absent", "x").unwrap(),
            Filter::is_null("absent").unwrap(),
            Filter::is_not_null("absent").unwrap(),
        ] {
            let expr: FilterExpression = filter.clone().into();
            assert!(
                !expr.matches(&record, &identity).unwrap(),
                "{} matched a missing field",
                filter.operator
            );
        }
        let exists: FilterExpression = Filter::exists("absent").unwrap().into();
        assert!(!exists.matches(&record, &identity).unwrap());
        let exists: FilterExpression = Filter::exists("present").unwrap().into();
        assert!(exists.matches(&record, &identity).unwrap());
    }

    #[test]
    fn null_is_distinguished_from_missing() {
        let record = json!({"a": null});
        let is_null: FilterExpression = Filter::is_null("a").unwrap().into();
        assert!(is_null.matches(&record, &identity).unwrap());
        let is_not_null: FilterExpression = Filter::is_not_null("a").unwrap().into();
        assert!(!is_not_null.matches(&record, &identity).unwrap());
        let exists: FilterExpression = Filter::exists("a").unwrap().into();
        assert!(exists.matches(&record, &identity).unwrap());
        // An ordered comparison against a present null is a non-match, not an
        // error.
        let gt: FilterExpression = Filter::gt("a", 0).unwrap().into();
        assert!(!gt.matches(&record, &identity).unwrap());
    }

    #[test]
    fn a_boolean_never_equals_a_number() {
        let record = json!({"flag": true});
        let eq_one: FilterExpression = Filter::eq("flag", 1).unwrap().into();
        assert!(!eq_one.matches(&record, &identity).unwrap());
        let eq_true: FilterExpression = Filter::eq("flag", true).unwrap().into();
        assert!(eq_true.matches(&record, &identity).unwrap());
    }

    #[test]
    fn an_integer_equals_the_same_float() {
        // The record round-tripped through JSON as an integer; the caller
        // wrote a float. `serde_json`'s own `==` says these differ.
        let record = json!({"score": 1});
        let expr: FilterExpression = Filter::eq("score", 1.0).unwrap().into();
        assert!(expr.matches(&record, &identity).unwrap());
        assert_ne!(json!(1), json!(1.0));
    }

    #[test]
    fn ordered_comparisons_cover_numbers_and_strings() {
        let record = json!({"n": 5, "s": "2026-09-14"});
        for (filter, want) in [
            (Filter::gt("n", 4).unwrap(), true),
            (Filter::gte("n", 5).unwrap(), true),
            (Filter::lt("n", 5).unwrap(), false),
            (Filter::lte("n", 5).unwrap(), true),
            (Filter::between("n", 1, 5).unwrap(), true),
            (Filter::between("n", 6, 9).unwrap(), false),
            (Filter::gt("s", "2026-01-01").unwrap(), true),
            (Filter::lt("s", "2026-01-01").unwrap(), false),
        ] {
            let expr: FilterExpression = filter.clone().into();
            assert_eq!(
                expr.matches(&record, &identity).unwrap(),
                want,
                "{} on {}",
                filter.operator,
                filter.field_name
            );
        }
    }

    #[test]
    fn an_incomparable_ordered_comparison_is_an_error() {
        let record = json!({"flag": true});
        let expr: FilterExpression = Filter::gt("flag", false).unwrap().into();
        let err = expr.matches(&record, &identity).unwrap_err().to_string();
        assert!(err.contains("incomparable"), "{err}");
    }

    #[test]
    fn membership_and_collection_operators() {
        let record = json!({"tag": "b", "tags": ["x", "y"]});
        let cases: Vec<(Filter, bool)> = vec![
            (
                Filter::any_of("tag", [json!("a"), json!("b")]).unwrap(),
                true,
            ),
            (Filter::none_of("tag", [json!("a")]).unwrap(), true),
            (Filter::contains("tags", "x").unwrap(), true),
            (Filter::contains("tags", "z").unwrap(), false),
            (
                Filter::contains_any("tags", [json!("z"), json!("y")]).unwrap(),
                true,
            ),
            (
                Filter::contains_all("tags", [json!("x"), json!("z")]).unwrap(),
                false,
            ),
            (
                Filter::contains_all("tags", [json!("x"), json!("y")]).unwrap(),
                true,
            ),
        ];
        for (filter, want) in cases {
            let expr: FilterExpression = filter.clone().into();
            assert_eq!(
                expr.matches(&record, &identity).unwrap(),
                want,
                "{} on {}",
                filter.operator,
                filter.field_name
            );
        }
    }

    #[test]
    fn text_operators() {
        let record = json!({"title": "quarterly report"});
        for (filter, want) in [
            (Filter::starts_with("title", "quarter").unwrap(), true),
            (Filter::ends_with("title", "report").unwrap(), true),
            (Filter::contains_text("title", "ly re").unwrap(), true),
            (Filter::contains_text("title", "annual").unwrap(), false),
        ] {
            let expr: FilterExpression = filter.into();
            assert_eq!(expr.matches(&record, &identity).unwrap(), want);
        }
    }

    #[test]
    fn groups_compose() {
        let record = json!({"a": 1, "b": 2});
        let both = FilterGroup::and(vec![
            Filter::eq("a", 1).unwrap().into(),
            Filter::eq("b", 2).unwrap().into(),
        ])
        .unwrap();
        assert!(both.matches(&record, &identity).unwrap());
        let either = FilterGroup::or(vec![
            Filter::eq("a", 99).unwrap().into(),
            Filter::eq("b", 2).unwrap().into(),
        ])
        .unwrap();
        assert!(either.matches(&record, &identity).unwrap());
        let negated = FilterGroup::not(Filter::eq("a", 1).unwrap().into()).unwrap();
        assert!(!negated.matches(&record, &identity).unwrap());
    }

    #[test]
    fn not_and_ne_differ_on_a_missing_field() {
        // The distinction the module docs call out, pinned: `ne` requires the
        // field to be present, `not(eq)` does not.
        let record = json!({"other": 1});
        let ne: FilterExpression = Filter::ne("a", 1).unwrap().into();
        assert!(!ne.matches(&record, &identity).unwrap());
        let not_eq = FilterGroup::not(Filter::eq("a", 1).unwrap().into()).unwrap();
        assert!(not_eq.matches(&record, &identity).unwrap());
    }

    #[test]
    fn the_resolver_maps_logical_names_onto_storage_names() {
        let record = json!({"stored_a": 1});
        let expr: FilterExpression = Filter::eq("a", 1).unwrap().into();
        assert!(expr
            .matches(&record, &|name| match name {
                "a" => Some("stored_a".into()),
                _ => None,
            })
            .unwrap());
    }

    #[test]
    fn an_undeclared_field_is_an_error_not_a_non_match() {
        let record = json!({"a": 1});
        let expr: FilterExpression = Filter::eq("nope", 1).unwrap().into();
        let err = expr
            .matches(&record, &|name| (name == "a").then(|| name.to_string()))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("not part of the collection definition"),
            "{err}"
        );
    }

    #[test]
    fn a_provider_operator_is_refused_by_the_portable_evaluator() {
        let record = json!({"a": 1});
        let filter = Filter::new(
            "a",
            FilterOperator::provider("azure_ai_search.match").unwrap(),
            Some(json!("x")),
        )
        .unwrap();
        let expr: FilterExpression = filter.into();
        let err = expr.matches(&record, &identity).unwrap_err().to_string();
        assert!(err.contains("provider-specific"), "{err}");
    }

    #[test]
    fn a_nested_path_is_refused_by_the_portable_evaluator() {
        let record = json!({"a": {"b": 1}});
        let expr: FilterExpression = Filter::eq("a.b", 1).unwrap().into();
        let err = expr.matches(&record, &identity).unwrap_err().to_string();
        assert!(err.contains("nested filter field paths"), "{err}");
    }
}

#[cfg(test)]
mod unvalidated_input_tests {
    use super::*;
    use serde_json::json;

    /// Every field resolves to itself.
    fn identity(name: &str) -> Option<String> {
        Some(name.to_string())
    }

    /// `matches` is public and `FilterExpression` is `Deserialize`, so a tree
    /// that never passed through a constructor can reach it. Indexing its
    /// children would then panic on caller data — in a method whose whole
    /// signature says it reports bad input as an error.
    #[test]
    fn an_unvalidated_expression_errors_rather_than_panicking() {
        let empty_not: FilterExpression =
            serde_json::from_value(json!({ "operator": "not", "filters": [] })).unwrap();
        assert!(empty_not.validate().is_err(), "validate already caught it");
        assert!(
            empty_not.matches(&json!({ "a": 1 }), &identity).is_err(),
            "and matches must not panic on the same input"
        );

        let short_between: FilterExpression = serde_json::from_value(
            json!({ "field_name": "a", "operator": "between", "value": [1] }),
        )
        .unwrap();
        assert!(short_between.validate().is_err());
        assert!(short_between
            .matches(&json!({ "a": 1 }), &identity)
            .is_err());
    }
}

#[cfg(test)]
mod numeric_precision_tests {
    use super::*;
    use serde_json::json;

    /// Routing every numeric comparison through `f64` — the obvious way to
    /// make `1` equal `1.0` — rounds past 2^53, so two adjacent 64-bit ids
    /// compare equal and a filter matches the wrong record.
    #[test]
    fn large_integers_compare_exactly() {
        let a = json!(9_007_199_254_740_992_i64);
        let b = json!(9_007_199_254_740_993_i64);
        assert_eq!(
            a.as_f64(),
            b.as_f64(),
            "the f64 round-trip really does collide"
        );

        assert!(!values_equal(&a, &b));
        assert_eq!(compare_values(&a, &b), Some(std::cmp::Ordering::Less));
        assert!(values_equal(&a, &json!(9_007_199_254_740_992_i64)));
    }

    /// …while the cross-type rule the module documents still holds.
    #[test]
    fn an_integer_still_equals_the_same_float() {
        assert!(values_equal(&json!(1), &json!(1.0)));
        assert_eq!(
            compare_values(&json!(2), &json!(1.5)),
            Some(std::cmp::Ordering::Greater)
        );
        assert!(!values_equal(&json!(1), &json!(true)));
    }

    /// The other half of the rounding problem: one integer operand and one
    /// float. Converting the *integer* to `f64` to compare them rounds it
    /// just the same, so the comparison goes the other way — a finite float's
    /// integer part is exact in `i128`, and its fraction breaks the tie.
    #[test]
    fn a_large_integer_does_not_equal_a_nearby_float() {
        let integer = json!(9_007_199_254_740_993_i64);
        let float = json!(9_007_199_254_740_992.0_f64);
        assert_eq!(
            integer.as_f64(),
            float.as_f64(),
            "the f64 round-trip really does collide"
        );

        assert!(!values_equal(&integer, &float));
        assert_eq!(
            compare_values(&integer, &float),
            Some(std::cmp::Ordering::Greater)
        );
        assert_eq!(
            compare_values(&float, &integer),
            Some(std::cmp::Ordering::Less)
        );
    }

    /// …and the fraction decides when the integer parts match.
    #[test]
    fn an_integer_orders_against_a_fractional_float() {
        assert_eq!(
            compare_values(&json!(3), &json!(3.5)),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            compare_values(&json!(3), &json!(2.5)),
            Some(std::cmp::Ordering::Greater)
        );
        assert_eq!(
            compare_values(&json!(-3), &json!(-2.5)),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            compare_values(&json!(3), &json!(3.0)),
            Some(std::cmp::Ordering::Equal)
        );
    }

    /// A float no integer can be compared against by truncation — an
    /// infinity, a NaN, or a value past `i128` — still answers sensibly.
    #[test]
    fn extreme_floats_are_handled() {
        assert_eq!(
            compare_values(&json!(1), &json!(f64::MAX)),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            compare_values(&json!(1), &json!(-f64::MAX)),
            Some(std::cmp::Ordering::Greater)
        );
        // `serde_json` cannot hold a NaN or an infinity in a `Number`, so the
        // only way one reaches the comparator is through a float operand the
        // caller built directly — which must not panic.
        assert!(compare_integer_to_float(1, f64::NAN).is_none());
        assert_eq!(
            compare_integer_to_float(1, f64::INFINITY),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            compare_integer_to_float(1, f64::NEG_INFINITY),
            Some(std::cmp::Ordering::Greater)
        );
    }

    /// A negative `i64` against a `u64` past `i64::MAX` has no common integer
    /// type; it must still order correctly rather than falling through.
    #[test]
    fn mixed_sign_integers_order_correctly() {
        let negative = json!(-1_i64);
        let huge = json!(u64::MAX);
        assert_eq!(
            compare_values(&negative, &huge),
            Some(std::cmp::Ordering::Less)
        );
        assert_eq!(
            compare_values(&huge, &negative),
            Some(std::cmp::Ordering::Greater)
        );
        assert!(!values_equal(&negative, &huge));
    }
}
