//@+leo-ver=5-thin
//@+node:ekr.20240928161210.1: * @file src/beautifier.rs
// tbo.rs

// From https://docs.rs/rustpython-parser/0.3.1/rustpython_parser/lexer/index.html

//@+<< beautifier.rs: suppressions >>
//@+node:ekr.20250117061304.1: ** << beautifier.rs: suppressions >>
// #! macros must be first.
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
// #![allow(unused_assignments)]
//@-<< beautifier.rs: suppressions >>

//@+<< beatufier.rs: crates and use >>
//@+node:ekr.20250117235612.1: ** << beatufier.rs: crates and use >>
extern crate rustpython_parser;
use rustpython_parser::{lexer::lex, Mode, Tok};
// use unicode_segmentation::UnicodeSegmentation;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path;
//@-<< beatufier.rs: crates and use >>

//@+others
//@+node:ekr.20250117091938.1: ** enum LexState
#[derive(Debug)]
enum LexState {
    NoState,
    Comment,
    FR,  // f-string or identifier.
    Identifier,
    Number,
    String1,
    String1Esc,
    String2,
    String2Esc,
    // TrippleString1,
    // TrippleString1Esc,
    // TrippleString2,
    // TrippleString2Esc,
    WSBlanks,
    WSTabs,
}
    
//@+node:ekr.20240929024648.120: ** struct InputTok
#[derive(Debug)]
struct InputTok<'a> {
    index: usize,  // index into slices.
    kind: &'a str,
    value: &'a str,
}
//@+node:ekr.20240929074037.1: ** class LeoBeautifier
#[derive(Debug)]
pub struct Beautifier {
    // Set in LB:beautify_one_file...
    args: Vec<String>,
    files_list: Vec<String>,
    stats: Stats,
    output_list: Vec<String>,
}

///// Temporary.
#[allow(dead_code)]
#[allow(non_snake_case)]
impl Beautifier {
    //@+others
    //@+node:ekr.20240929074037.114: *3*  LB.new
    pub fn new() -> Beautifier {
        let mut x = Beautifier {
            // Set in beautify_one_file
            args: Vec::new(),
            files_list: Vec::new(),
            output_list: Vec::new(),
            stats: Stats::new(),
        };
        x.get_args();
        return x;
    }
    //@+node:ekr.20240929074037.2: *3* LB.add_output_string
    #[allow(unused_variables)]
    fn add_output_string(&mut self, kind: &str, value: &str) {
        //! Add value to the output list.
        //! kind is for debugging.
        if !value.is_empty() {
            self.output_list.push(value.to_string())
        }
    }
    //@+node:ekr.20240929074037.4: *3* LB.beautify_all_files
    pub fn beautify_all_files(&mut self) {
        // for file_name in self.files_list.clone() {
        for file_name in self.files_list.clone() {
            self.beautify_one_file(&file_name);
        }
    }

    //@+node:ekr.20240929074037.5: *3* LB.beautify_one_file
    fn beautify_one_file(&mut self, file_name: &str) {
        self.stats.n_files += 1;
        if true {
            // Testing only: print the short file name.
            let file_path = path::Path::new(file_name);
            let os_str = file_path.file_name().unwrap(); // &OsStr
            let short_file_name = os_str.to_str().unwrap();
            println!("{short_file_name}");
        }
        // Read the file into contents (a String).
        let t1 = std::time::Instant::now();
        let contents = fs::read_to_string(file_name).expect("Error reading{file_name}");
        self.stats.read_time += t1.elapsed().as_nanos();

        // Create (an immutable!) list of input tokens.
        let t2 = std::time::Instant::now();
        let _input_tokens = self.make_prototype_input_list(&contents);
        // let _input_tokens = self.make_input_list(&contents);
        self.stats.make_tokens_time += t2.elapsed().as_nanos();

        // ***
            // let input_tokens = self.make_input_list(&contents);
            // self.stats.make_tokens_time += t2.elapsed().as_nanos();
            // // Annotate tokens (the prepass).
            // let t3 = std::time::Instant::now();
            // let mut annotator = Annotator::new(&input_tokens);
            // let annotated_tokens = annotator.annotate();
            // self.stats.annotation_time += t3.elapsed().as_nanos();
            // // Beautify.
            // let t4 = std::time::Instant::now();
            // self.beautify(&annotated_tokens);
            // self.stats.beautify_time += t4.elapsed().as_nanos();
    }
    //@+node:ekr.20240929074037.7: *3* LB.do_*
    //@+node:ekr.20241002071143.1: *4* tbo.do_ws
    // *** Temporary
    #[allow(unused_variables)]
    fn do_ws(&mut self, kind: &str, value: &str) {
        //! Handle the "ws" pseudo-token.
        //! Put the whitespace only if if ends with backslash-newline.

        // To do.

        // let last_token = self.input_tokens[self.index - 1];
        // let is_newline = kind in ("nl", "newline");
        // if is_newline {
        // self.pending_lws = val;
        // self.pending_ws = "";
        // }
        // else if "\\\n" in val {
        // self.pending_lws = "";
        // self.pending_ws = val;
        // }
        // else {
        // self.pending_ws = val
        // }
    }
    //@+node:ekr.20240929074037.8: *4* LB:Handlers with values
    //@+node:ekr.20240929074037.9: *5* LB.do_Comment
    fn do_Comment(&mut self, tok_value: &str) {
        // print!("{tok_value}");  // Correct.
        // print!("{value} ");  // Wrong!
        self.add_output_string("Comment", tok_value);
    }
    //@+node:ekr.20240929074037.10: *5* LB.do_Complex
    fn do_Complex(&mut self, tok_value: &str) {
        self.add_output_string("Complex", tok_value);
    }
    //@+node:ekr.20240929074037.11: *5* LB.do_Float
    fn do_Float(&mut self, tok_value: &str) {
        self.add_output_string("Float", tok_value);
    }
    //@+node:ekr.20240929074037.12: *5* LB.do_Int
    fn do_Int(&mut self, tok_value: &str) {
        self.add_output_string("Int", tok_value);
    }
    //@+node:ekr.20240929074037.13: *5* LB.do_Name
    fn do_Name(&mut self, tok_value: &str) {
        self.add_output_string("Name", tok_value);
    }
    //@+node:ekr.20240929074037.14: *5* LB.do_String
    fn do_String(&mut self, tok_value: &str) {
        // correct.
        // print!("{tok_value}");

        // incorrect.
        // let quote = if *triple_quoted {"'''"} else {"'"};
        // print!("{:?}:{quote}{value}{quote}", kind);

        self.add_output_string("String", tok_value);
    }
    //@+node:ekr.20240929074037.15: *4* LB:Handlers using lws
    //@+node:ekr.20240929074037.16: *5* LB.do_Dedent
    fn do_Dedent(&mut self, tok_value: &str) {
        self.add_output_string("Dedent", tok_value);
    }
    //@+node:ekr.20240929074037.17: *5* LB.do_Indent
    fn do_Indent(&mut self, tok_value: &str) {
        self.add_output_string("Indent", tok_value);
    }
    //@+node:ekr.20240929074037.18: *5* LB.do_Newline
    fn do_Newline(&mut self) {
        self.add_output_string("Indent", "\n");
    }
    //@+node:ekr.20240929074037.19: *5* LB.do_NonLogicalNewline
    fn do_NonLogicalNewline(&mut self) {
        self.add_output_string("Indent", "\n");
    }
    //@+node:ekr.20240929074037.20: *4* LB:Handlers w/o values
    //@+node:ekr.20240929074037.21: *5* LB.do_Amper
    fn do_Amper(&mut self) {
        self.add_output_string("Amper", "&");
    }
    //@+node:ekr.20240929074037.22: *5* LB.do_AmperEqual
    fn do_AmperEqual(&mut self) {
        self.add_output_string("AmperEqual", "&=");
    }
    //@+node:ekr.20240929074037.23: *5* LB.do_And
    fn do_And(&mut self) {
        self.add_output_string("And", "and");
    }
    //@+node:ekr.20240929074037.24: *5* LB.do_As
    fn do_As(&mut self) {
        self.add_output_string("As", "as");
    }
    //@+node:ekr.20240929074037.25: *5* LB.do_Assert
    fn do_Assert(&mut self) {
        self.add_output_string("Assert", "assert");
    }
    //@+node:ekr.20240929074037.26: *5* LB.do_Async
    fn do_Async(&mut self) {
        self.add_output_string("Async", "async");
    }
    //@+node:ekr.20240929074037.27: *5* LB.do_At
    fn do_At(&mut self) {
        self.add_output_string("At", "@");
    }
    //@+node:ekr.20240929074037.28: *5* LB.do_AtEqual
    fn do_AtEqual(&mut self) {
        self.add_output_string("AtEqual", "@=");
    }
    //@+node:ekr.20240929074037.29: *5* LB.do_Await
    fn do_Await(&mut self) {
        self.add_output_string("Await", "await");
    }
    //@+node:ekr.20240929074037.30: *5* LB.do_Break
    fn do_Break(&mut self) {
        self.add_output_string("Break", "break");
    }
    //@+node:ekr.20240929074037.31: *5* LB.do_Case
    fn do_Case(&mut self) {
        self.add_output_string("Case", "case");
    }
    //@+node:ekr.20240929074037.32: *5* LB.do_CircumFlex
    fn do_CircumFlex(&mut self) {
        self.add_output_string("CircumFlex", "^");
    }
    //@+node:ekr.20240929074037.33: *5* LB.do_CircumflexEqual
    fn do_CircumflexEqual(&mut self) {
        self.add_output_string("CircumflexEqual", "^=");
    }
    //@+node:ekr.20240929074037.34: *5* LB.do_Class
    fn do_Class(&mut self) {
        self.add_output_string("Class", "class");
    }
    //@+node:ekr.20240929074037.35: *5* LB.do_Colon
    fn do_Colon(&mut self) {
        self.add_output_string("Colon", ":");
    }
    //@+node:ekr.20240929074037.36: *5* LB.do_ColonEqual
    fn do_ColonEqual(&mut self) {
        self.add_output_string("ColonEqual", ":=");
    }
    //@+node:ekr.20240929074037.37: *5* LB.do_Comma
    fn do_Comma(&mut self) {
        self.add_output_string("Comma", ",");
    }
    //@+node:ekr.20240929074037.38: *5* LB.do_Continue
    fn do_Continue(&mut self) {
        self.add_output_string("Continue", "continue");
    }
    //@+node:ekr.20240929074037.39: *5* LB.do_Def
    fn do_Def(&mut self) {
        self.add_output_string("Def", "def");
    }
    //@+node:ekr.20240929074037.40: *5* LB.do_Del
    fn do_Del(&mut self) {
        self.add_output_string("Del", "del");
    }
    //@+node:ekr.20240929074037.41: *5* LB.do_Dot
    fn do_Dot(&mut self) {
        self.add_output_string("Dot", ".");
    }
    //@+node:ekr.20240929074037.42: *5* LB.do_DoubleSlash
    fn do_DoubleSlash(&mut self) {
        self.add_output_string("DoubleSlash", "//");
    }
    //@+node:ekr.20240929074037.43: *5* LB.do_DoubleSlashEqual
    fn do_DoubleSlashEqual(&mut self) {
        self.add_output_string("DoubleSlashEqual", "//=");
    }
    //@+node:ekr.20240929074037.44: *5* LB.do_DoubleStar
    fn do_DoubleStar(&mut self) {
        self.add_output_string("DoubleStar", "**");
    }
    //@+node:ekr.20240929074037.45: *5* LB.do_DoubleStarEqual
    fn do_DoubleStarEqual(&mut self) {
        self.add_output_string("DoubleStarEqual", "**=");
    }
    //@+node:ekr.20240929074037.46: *5* LB.do_Elif
    fn do_Elif(&mut self) {
        self.add_output_string("Elif", "elif");
    }
    //@+node:ekr.20240929074037.47: *5* LB.do_Ellipsis
    fn do_Ellipsis(&mut self) {
        self.add_output_string("Ellipsis", "...");
    }
    //@+node:ekr.20240929074037.48: *5* LB.do_Else
    fn do_Else(&mut self) {
        self.add_output_string("Else", "else");
    }
    //@+node:ekr.20240929074037.49: *5* LB.do_EndOfFile
    fn do_EndOfFile(&mut self) {
        self.add_output_string("EndOfFile", "EOF");
    }
    //@+node:ekr.20240929074037.50: *5* LB.do_EqEqual
    fn do_EqEqual(&mut self) {
        self.add_output_string("EqEqual", "==");
    }
    //@+node:ekr.20240929074037.51: *5* LB.do_Equal
    fn do_Equal(&mut self) {
        self.add_output_string("Equal", "=");
    }
    //@+node:ekr.20240929074037.52: *5* LB.do_Except
    fn do_Except(&mut self) {
        self.add_output_string("Except", "except");
    }
    //@+node:ekr.20240929074037.53: *5* LB.do_False
    fn do_False(&mut self) {
        self.add_output_string("False", "False");
    }
    //@+node:ekr.20240929074037.54: *5* LB.do_Finally
    fn do_Finally(&mut self) {
        self.add_output_string("Finally", "finally");
    }
    //@+node:ekr.20240929074037.55: *5* LB.do_For
    fn do_For(&mut self) {
        self.add_output_string("For", "for");
    }
    //@+node:ekr.20240929074037.56: *5* LB.do_From
    fn do_From(&mut self) {
        self.add_output_string("From", "from");
    }
    //@+node:ekr.20240929074037.57: *5* LB.do_Global
    fn do_Global(&mut self) {
        self.add_output_string("Global", "global");
    }
    //@+node:ekr.20240929074037.58: *5* LB.do_Greater
    fn do_Greater(&mut self) {
        self.add_output_string("Greater", ">");
    }
    //@+node:ekr.20240929074037.59: *5* LB.do_GreaterEqual
    fn do_GreaterEqual(&mut self) {
        self.add_output_string("GreaterEqual", ">-");
    }
    //@+node:ekr.20240929074037.60: *5* LB.do_If
    fn do_If(&mut self) {
        self.add_output_string("If", "if");
    }
    //@+node:ekr.20240929074037.61: *5* LB.do_Import
    fn do_Import(&mut self) {
        self.add_output_string("Import", "import");
    }
    //@+node:ekr.20240929074037.62: *5* LB.do_In
    fn do_In(&mut self) {
        self.add_output_string("In", "in");
    }
    //@+node:ekr.20240929074037.63: *5* LB.do_Is
    fn do_Is(&mut self) {
        self.add_output_string("Is", "is");
    }
    //@+node:ekr.20240929074037.64: *5* LB.do_Lambda
    fn do_Lambda(&mut self) {
        self.add_output_string("Lambda", "lambda");
    }
    //@+node:ekr.20240929074037.65: *5* LB.do_Lbrace
    fn do_Lbrace(&mut self) {
        self.add_output_string("Lbrace", "[");
    }
    //@+node:ekr.20240929074037.66: *5* LB.do_LeftShift
    fn do_LeftShift(&mut self) {
        self.add_output_string("LeftShift", "<<");
    }
    //@+node:ekr.20240929074037.67: *5* LB.do_LeftShiftEqual
    fn do_LeftShiftEqual(&mut self) {
        self.add_output_string("LeftShiftEqual", "<<=");
    }
    //@+node:ekr.20240929074037.68: *5* LB.do_Less
    fn do_Less(&mut self) {
        self.add_output_string("Less", "<");
    }
    //@+node:ekr.20240929074037.69: *5* LB.do_LessEqual
    fn do_LessEqual(&mut self) {
        self.add_output_string("LessEqual", "<=");
    }
    //@+node:ekr.20240929074037.70: *5* LB.do_Lpar
    fn do_Lpar(&mut self) {
        self.add_output_string("Lpar", "(");
    }
    //@+node:ekr.20240929074037.71: *5* LB.do_Lsqb
    fn do_Lsqb(&mut self) {
        self.add_output_string("Lsqb", "[");
    }
    //@+node:ekr.20240929074037.72: *5* LB.do_Match
    fn do_Match(&mut self) {
        self.add_output_string("Match", "match");
    }
    //@+node:ekr.20240929074037.73: *5* LB.do_Minus
    fn do_Minus(&mut self) {
        self.add_output_string("Minus", "-");
    }
    //@+node:ekr.20240929074037.74: *5* LB.do_MinusEqual
    fn do_MinusEqual(&mut self) {
        self.add_output_string("MinusEqual", "-=");
    }
    //@+node:ekr.20240929074037.75: *5* LB.do_None
    fn do_None(&mut self) {
        self.add_output_string("None", "None");
    }
    //@+node:ekr.20240929074037.76: *5* LB.do_Nonlocal
    fn do_Nonlocal(&mut self) {
        self.add_output_string("Nonlocal", "nonlocal");
    }
    //@+node:ekr.20240929074037.77: *5* LB.do_Not
    fn do_Not(&mut self) {
        self.add_output_string("Not", "not");
    }
    //@+node:ekr.20240929074037.78: *5* LB.do_NotEqual
    fn do_NotEqual(&mut self) {
        self.add_output_string("NotEqual", "!=");
    }
    //@+node:ekr.20240929074037.79: *5* LB.do_Or
    fn do_Or(&mut self) {
        self.add_output_string("Or", "or");
    }
    //@+node:ekr.20240929074037.80: *5* LB.do_Pass
    fn do_Pass(&mut self) {
        self.add_output_string("Pass", "pass");
    }
    //@+node:ekr.20240929074037.81: *5* LB.do_Percent
    fn do_Percent(&mut self) {
        self.add_output_string("Percent", "%");
    }
    //@+node:ekr.20240929074037.82: *5* LB.do_PercentEqual
    fn do_PercentEqual(&mut self) {
        self.add_output_string("PercentEqual", "%=");
    }
    //@+node:ekr.20240929074037.83: *5* LB.do_Plus
    fn do_Plus(&mut self) {
        self.add_output_string("Plus", "+");
    }
    //@+node:ekr.20240929074037.84: *5* LB.do_PlusEqual
    fn do_PlusEqual(&mut self) {
        self.add_output_string("PlusEqual", "+=");
    }
    //@+node:ekr.20240929074037.85: *5* LB.do_Raise
    fn do_Raise(&mut self) {
        self.add_output_string("Raise", "raise");
    }
    //@+node:ekr.20240929074037.86: *5* LB.do_Rarrow
    fn do_Rarrow(&mut self) {
        self.add_output_string("Rarrow", "->");
    }
    //@+node:ekr.20240929074037.87: *5* LB.do_Rbrace
    fn do_Rbrace(&mut self) {
        self.add_output_string("Rbrace", "]");
    }
    //@+node:ekr.20240929074037.88: *5* LB.do_Return
    fn do_Return(&mut self) {
        self.add_output_string("Return", "return");
    }
    //@+node:ekr.20240929074037.89: *5* LB.do_RightShift
    fn do_RightShift(&mut self) {
        self.add_output_string("RightShift", ">>");
    }
    //@+node:ekr.20240929074037.90: *5* LB.do_RightShiftEqual
    fn do_RightShiftEqual(&mut self) {
        self.add_output_string("RightShiftEqual", ">>=");
    }
    //@+node:ekr.20240929074037.91: *5* LB.do_Rpar
    fn do_Rpar(&mut self) {
        self.add_output_string("Rpar", ")");
    }
    //@+node:ekr.20240929074037.92: *5* LB.do_Rsqb
    fn do_Rsqb(&mut self) {
        self.add_output_string("Rsqb", "]");
    }
    //@+node:ekr.20240929074037.93: *5* LB.do_Semi
    fn do_Semi(&mut self) {
        self.add_output_string("Semi", ";");
    }
    //@+node:ekr.20240929074037.94: *5* LB.do_Slash
    fn do_Slash(&mut self) {
        self.add_output_string("Slash", "/");
    }
    //@+node:ekr.20240929074037.95: *5* LB.do_SlashEqual
    fn do_SlashEqual(&mut self) {
        self.add_output_string("SlashEqual", "/=");
    }
    //@+node:ekr.20240929074037.96: *5* LB.do_Star
    fn do_Star(&mut self) {
        self.add_output_string("Star", "*");
    }
    //@+node:ekr.20240929074037.97: *5* LB.do_StarEqual
    fn do_StarEqual(&mut self) {
        self.add_output_string("StarEqual", "*=");
    }
    //@+node:ekr.20240929074037.98: *5* LB.do_StartExpression
    fn do_StartExpression(&mut self) {
        // self.add_output_string("StartExpression", "");
    }
    //@+node:ekr.20240929074037.99: *5* LB.do_StartInteractive
    fn do_StartInteractive(&mut self) {
        // self.add_output_string("StartModule", "");
    }
    //@+node:ekr.20240929074037.100: *5* LB.do_StarModule
    fn do_StartModule(&mut self) {
        // self.add_output_string("StartModule", "");
        println!("do_StartModule");
    }
    //@+node:ekr.20240929074037.101: *5* LB.do_Tilde
    fn do_Tilde(&mut self) {
        self.add_output_string("Tilde", "~");
    }
    //@+node:ekr.20240929074037.102: *5* LB.do_True
    fn do_True(&mut self) {
        self.add_output_string("True", "True");
    }
    //@+node:ekr.20240929074037.103: *5* LB.do_Try
    fn do_Try(&mut self) {
        self.add_output_string("Try", "try");
    }
    //@+node:ekr.20240929074037.104: *5* LB.do_Type
    fn do_Type(&mut self) {
        self.add_output_string("Type", "type");
    }
    //@+node:ekr.20240929074037.105: *5* LB.do_Vbar
    fn do_Vbar(&mut self) {
        self.add_output_string("Vbar", "|");
    }
    //@+node:ekr.20240929074037.106: *5* LB.do_VbarEqual
    fn do_VbarEqual(&mut self) {
        self.add_output_string("VbarEqual", "|=");
    }
    //@+node:ekr.20240929074037.107: *5* LB.do_While
    fn do_While(&mut self) {
        self.add_output_string("While", "while");
    }
    //@+node:ekr.20240929074037.108: *5* LB.do_With
    fn do_With(&mut self) {
        self.add_output_string("With", "with");
    }
    //@+node:ekr.20240929074037.109: *5* LB.do_Yield
    fn do_Yield(&mut self) {
        self.add_output_string("Yield", "yield");
    }
    //@+node:ekr.20240929074037.110: *3* LB.enabled
    fn enabled(&self, arg: &str) -> bool {
        //! Beautifier::enabled: return true if the given command-line argument is enabled.
        //! Example:  x.enabled("--report");
        return self.args.contains(&arg.to_string());
    }
    //@+node:ekr.20240929074037.111: *3* LB.get_args
    fn get_args(&mut self) {
        //! Beautifier::get_args: Set the args and files_list ivars.
        let args: Vec<String> = env::args().collect();
        let valid_args = vec![
            "--all",
            "--beautified",
            "--diff",
            "-h",
            "--help",
            "--report",
            "--write",
        ];
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                if valid_args.contains(&arg.as_str()) {
                    self.args.push(arg.to_string())
                } else if arg.as_str().starts_with("--") || arg.as_str().starts_with("--") {
                    println!("Ignoring invalid arg: {arg}");
                } else {
                    println!("File: {arg}");
                    self.files_list.push(arg.to_string());
                }
            }
        }
    }
    //@+node:ekr.20250116134245.1: *3* LB.make_prototype_input_list
    /// make_prototype_input_list
    /// w/o graphemes:
    ///     w/  result_push: 1.08 ms to 1.35 ms. (Old: 2.95 to 3.30 ms.)
    ///     w/o result_push: 0.72 ms to 0.90 ms.
    /// w graphemes:
    ///     w/o result_push: 11.09 ms to 12.05 ms.

    fn make_prototype_input_list<'a>(&mut self, contents: &'a str) -> Vec<InputTok<'a>> {

        // Stats.
        let mut line_number: usize = 0;
        let mut n_tokens: u64 = 0;
        let mut n_ws_tokens: u64 = 0;
        
        // The main loop.
        let mut state: LexState = LexState::NoState;
        let mut result: Vec<InputTok> = Vec::new();
        let mut index: usize = 0;
        let mut start_index: usize = 0;
        
        for ch in contents.chars() {
            index += 1;
            if ch == '\r' {
                index += 1;
                continue;
            }
            if ch == '\n'  {
                    line_number += 1;
            }
            // println!("line: {line_number} {start_index}..{index} ch: {ch:?} state: {state:?}");
            use LexState::*;
            match &state {
                NoState => {
                    match ch {
                        '\n' => {
                            n_tokens += 1;
                            result.push(
                                InputTok{index: start_index, kind: &"newline", value: &contents[start_index..index]}
                            );
                            start_index = index;
                        },
                        '#' => {
                             state = Comment;
                        },
                        ' ' => {
                            state = WSBlanks;
                        },
                        '\t' => {
                            state = WSTabs;
                        },
                        '0'..'9' => {
                            state = Number;
                        },
                        '\'' => {
                            state = String1;
                        },
                        '"' => {
                            state = String2;
                        },
                        'A'..='Z' | 'a'..='z' | '_' => {
                            state = Identifier;
                        },
                        _ => {},
                    }
                },
                Comment => {
                    match ch {
                        '\n' => {
                            n_tokens += 1;
                            // *** result.push(InputTok{kind: &"comment", value: &String::from(token)});
                            start_index = index;
                            state = NoState;
                        },
                        _ => {},
                    }
                }
                FR => {}, // f-string or identifier.
                Identifier => {
                    match ch {
                        'A'..='Z' | 'a'..='z' | '_' => {},
                        _ => {
                            n_tokens += 1;
                            // *** result.push(InputTok{kind: &"identifier", value: &String::from(token)});
                            state = NoState;
                        },
                    }
                },
                Number => {
                    match ch {
                        // "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" => {
                        '0'..='9' => {},
                        _ => {
                            n_tokens += 1;
                            // *** result.push(InputTok{kind: &"number", value: &String::from(token)});
                            start_index = index;
                            state = NoState;
                        },
                    }
                }
                String1 => {
                    // This is a *Python* single-quoted string.
                    match ch {
                        '\\' => {
                            state = String1Esc;
                        },
                        '\'' => {
                            n_tokens += 1;
                            // *** result.push(InputTok{kind: &"string", value: &String::from(token)});
                            start_index = index;
                            state = NoState;
                        },
                        _ => {},
                    }
                },
                String2 => {
                    match ch {
                        '\\' => {
                            state = String2Esc;
                        },
                        '"' => {
                            // *** result.push(InputTok{kind: &"string", value: &String::from(token)});
                            n_tokens += 1;
                            start_index = index;
                            state = NoState;
                        },
                        _ => {},
                    }
                },  
                String1Esc => {
                    state = String1;
                },
                String2Esc => {
                    state = String2;
                },
                WSBlanks => {
                    if ch != ' ' {
                        n_tokens += 1;
                        n_ws_tokens += 1;
                        // *** result.push(InputTok{kind: &"ws", value: &String::from(token)});
                        start_index = index;
                        state = NoState;
                    }
                },  
                WSTabs => {
                    if ch != '\t' {
                        n_tokens += 1;
                        n_ws_tokens += 1;
                        // *** result.push(InputTok{kind: &"ws", value: &String::from(token)});
                        start_index = index;
                        state = NoState;
                    }
                },  
            }
        }

        println!("lines: {line_number}");
        // println!("state: {state:?}");

        // Update counts.
        self.stats.n_tokens += n_tokens;
        self.stats.n_ws_tokens += n_ws_tokens;
        return result;
    }
    //@+node:ekr.20240929074037.115: *3* LB.show_args
    fn show_args(&self) {
        println!("Command-line arguments...");
        for (i, arg) in self.args.iter().enumerate() {
            if i > 0 {
                println!("  {arg}");
            }
        }
        for file_arg in self.files_list.iter() {
            println!("  {file_arg}");
        }
    }
    //@+node:ekr.20240929074037.116: *3* LB.show_help
    fn show_help(&self) {
        //! Beautifier::show_help: print the help messages.
        println!(
            "{}",
            textwrap::dedent(
                "
            Beautify or diff files.

            -h --help:      Print this help message and exit.
            --all:          Beautify all files, even unchanged files.
            --beautified:   Report beautified files individually, even if not written.
            --diff:         Show diffs instead of changing files.
            --report:       Print summary report.
            --write:        Write beautifed files (dry-run mode otherwise).
        "
            )
        );
    }
    //@-others
}
//@+node:ekr.20240929074547.1: ** class Stats
// Allow unused write_time
#[allow(dead_code)]
#[derive(Debug)]
pub struct Stats {
    // Cumulative statistics for all files.
    n_files: u64,     // Number of files.
    n_tokens: u64,    // Number of tokens.
    n_ws_tokens: u64, // Number of pseudo-ws tokens.

    // Timing stat, in microseconds...
    annotation_time: u128,
    beautify_time: u128,
    make_tokens_time: u128,
    read_time: u128,
    write_time: u128,
}

// Calling Stats.report is optional.
#[allow(dead_code)]
impl Stats {
    //@+others
    //@+node:ekr.20241001100954.1: *3*  Stats::new
    pub fn new() -> Stats {
        let x = Stats {
            // Cumulative counts.
            n_files: 0,     // Number of files.
            n_tokens: 0,    // Number of tokens.
            n_ws_tokens: 0, // Number of pseudo-ws tokens.

            // Timing stats, in nanoseconds...
            annotation_time: 0,
            beautify_time: 0,
            make_tokens_time: 0,
            read_time: 0,
            write_time: 0,
        };
        return x;
    }
    //@+node:ekr.20240929080242.1: *3* Stats::fmt_ns
    fn fmt_ns(&mut self, t: u128) -> String {
        //! Convert nanoseconds to fractional milliseconds.
        let ms = t / 1000000;
        let micro = (t % 1000000) / 10000; // 2-places only.
                                           // println!("t: {t:8} ms: {ms:03} micro: {micro:02}");
        return f!("{ms:4}.{micro:02}");
    }

    //@+node:ekr.20240929075236.1: *3* Stats::report
    fn report(&mut self) {
        // Cumulative counts.
        let n_files = self.n_files;
        let n_tokens = self.n_tokens;
        let n_ws_tokens = self.n_ws_tokens;
        // Print cumulative timing stats, in ms.
        let annotation_time = self.fmt_ns(self.annotation_time);
        let beautify_time = self.fmt_ns(self.beautify_time);
        let make_tokens_time = self.fmt_ns(self.make_tokens_time);
        let read_time = self.fmt_ns(self.read_time);
        let write_time = self.fmt_ns(self.write_time);
        let total_time_ns = self.annotation_time
            + self.beautify_time
            + self.make_tokens_time
            + self.read_time
            + self.write_time;
        let total_time = self.fmt_ns(total_time_ns);
        println!("");
        println!("     files: {n_files}, tokens: {n_tokens}, ws tokens: {n_ws_tokens}");
        println!("       read: {read_time:>7} ms");
        println!("make_tokens: {make_tokens_time:>7} ms");
        println!("   annotate: {annotation_time:>7} ms");
        println!("   beautify: {beautify_time:>7} ms");
        println!("      write: {write_time:>7} ms");
        println!("      total: {total_time:>7} ms");
    }
    //@-others
}
//@+node:ekr.20241003093554.1: ** fn: entry
pub fn entry() {
    main();
}
//@+node:ekr.20241005091217.1: ** fn: is_python_keyword (to do)
// def is_python_keyword(self, token: Optional[InputToken]) -> bool:
// """Return True if token is a 'name' token referring to a Python keyword."""
// if not token or token.kind != 'name':
// return False
// return keyword.iskeyword(token.value) or keyword.issoftkeyword(token.value)

// Keywords:
// False      await      else       import     pass
// None       break      except     in         raise
// True       class      finally    is         return
// and        continue   for        lambda     try
// as         def        from       nonlocal   while
// assert     del        global     not        with
// async      elif       if         or         yield

// Soft keywords:
// match, case, type and _

// *** Remove leading underscores.
fn is_python_keyword(_token: &InputTok) -> bool {
    return false;

    // *** Not ready yet.
    // //! Return True if token is a 'name' token referring to a Python keyword.
    // if token.kind != "name" {
    // return false;
    // }
    // // let word = &token.value;  // &String
    // return false;  // ***
}
//@+node:ekr.20241005092549.1: ** fn: is_unary_op_with_prev (to do)
// def is_unary_op_with_prev(self, prev: Optional[InputToken], token: InputToken) -> bool:
// """
// Return True if token is a unary op in the context of prev, the previous
// significant token.
// """
// if token.value == '~':  # pragma: no cover
// return True
// if prev is None:
// return True  # pragma: no cover
// assert token.value in '**-+', repr(token.value)
// if prev.kind in ('number', 'string'):
// return_val = False
// elif prev.kind == 'op' and prev.value in ')]':
// # An unnecessary test?
// return_val = False  # pragma: no cover
// elif prev.kind == 'op' and prev.value in '{([:,':
// return_val = True
// elif prev.kind != 'name':
// # An unnecessary test?
// return_val = True  # pragma: no cover
// else:
// # prev is a'name' token.
// return self.is_python_keyword(token)
// return return_val

// *** Remove leading underscores.
fn is_unary_op_with_prev(_prev_token: &InputTok, _token: &InputTok) -> bool {
    return false; // ***
}
//@+node:ekr.20241003093722.1: ** fn: main
//@@language rust
pub fn main() {
    // Main line of beautifier.
    let mut x = Beautifier::new();
    if true {
        // testing.
        println!("");
        for file_path in [
            // "C:\\Repos\\ekr-tbo-in-rust\\test\\test1.py",
            "C:\\Repos\\leo-editor\\leo\\core\\leoFrame.py",
            // "C:\\Repos\\leo-editor\\leo\\core\\leoApp.py"
        ] {
            x.beautify_one_file(&file_path);
        }
        x.stats.report();
    } else {
        if x.enabled("--help") || x.enabled("-h") {
            x.show_help();
            return;
        }
        x.show_args();
        x.beautify_all_files();
    }
}
//@-others

//@-leo
