use ruff_formatter::FormatRuleWithOptions;
use ruff_python_ast::StringLiteral;

use crate::prelude::*;
use crate::preview::is_f_string_implicit_concatenated_string_literal_quotes_enabled;
use crate::string::{docstring, Quoting, StringNormalizer};
use crate::QuoteStyle;

#[derive(Default)]
pub struct FormatStringLiteral {
    layout: StringLiteralKind,
}

impl FormatRuleWithOptions<StringLiteral, PyFormatContext<'_>> for FormatStringLiteral {
    type Options = StringLiteralKind;

    fn with_options(mut self, layout: StringLiteralKind) -> Self {
        self.layout = layout;
        self
    }
}

/// The kind of a string literal.
#[derive(Copy, Clone, Debug, Default)]
pub enum StringLiteralKind {
    /// A normal string literal e.g., `"foo"`.
    #[default]
    String,
    /// A string literal used as a docstring.
    Docstring,
    /// A string literal that is implicitly concatenated with an f-string. This
    /// makes the overall expression an f-string whose quoting detection comes
    /// from the parent node (f-string expression).
    #[deprecated]
    #[allow(private_interfaces)]
    InImplicitlyConcatenatedFString(Quoting),
}

impl StringLiteralKind {
    /// Checks if this string literal is a docstring.
    pub(crate) const fn is_docstring(self) -> bool {
        matches!(self, StringLiteralKind::Docstring)
    }

    /// Returns the quoting to be used for this string literal.
    fn quoting(self, context: &PyFormatContext) -> Quoting {
        match self {
            StringLiteralKind::String | StringLiteralKind::Docstring => Quoting::CanChange,
            #[allow(deprecated)]
            StringLiteralKind::InImplicitlyConcatenatedFString(quoting) => {
                // Allow string literals to pick the "optimal" quote character
                // even if any other fstring in the implicit concatenation uses an expression
                // containing a quote character.
                // TODO: Remove StringLiteralKind::InImplicitlyConcatenatedFString when promoting
                //   this style to stable and remove the layout from `AnyStringPart::String`.
                if is_f_string_implicit_concatenated_string_literal_quotes_enabled(context) {
                    Quoting::CanChange
                } else {
                    quoting
                }
            }
        }
    }
}

impl FormatNodeRule<StringLiteral> for FormatStringLiteral {
    fn fmt_fields(&self, item: &StringLiteral, f: &mut PyFormatter) -> FormatResult<()> {
        let quote_style = f.options().quote_style();
        let quote_style = if self.layout.is_docstring() && !quote_style.is_preserve() {
            // Per PEP 8 and PEP 257, always prefer double quotes for docstrings,
            // except when using quote-style=preserve
            QuoteStyle::Double
        } else {
            quote_style
        };

        let normalized = StringNormalizer::from_context(f.context())
            .with_quoting(self.layout.quoting(f.context()))
            .with_preferred_quote_style(quote_style)
            .normalize(item.into());

        if self.layout.is_docstring() {
            docstring::format(&normalized, f)
        } else {
            normalized.fmt(f)
        }
    }
}