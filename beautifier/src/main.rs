//@+leo-ver=5-thin
//@+node:ekr.20240927151332.1: * @file src/main.rs
//@@language rust
// main.rs

// extern crates must be in crate root.

// For the f! macro.
#[macro_use]
extern crate fstrings;

mod beautifier;

fn main() {
    beautifier::entry();
}
//@-leo
