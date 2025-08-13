//! Build script for generating gRPC service code from protocol buffers
//! 
//! Generates Rust code for:
//! - Discovery service (job submission, status, results)
//! - Fingerprint service (asset fingerprinting)
//! - Risk tagging service (vulnerability risk assessment)
//! - Audit service (security audit trail)

use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?); 
    
    // Configure tonic to generate server and client code
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir(&out_dir)
        .compile(
            &[
                "../../proto/xorb/discovery/v1/discovery.proto",
                "../../proto/xorb/fingerprint/v1/fingerprint.proto", 
                "../../proto/xorb/risktag/v1/risktag.proto",
                "../../proto/xorb/audit/v1/audit.proto",
            ],
            &["../../proto"]
        )?;
        
    println!("cargo:rerun-if-changed=../../proto");
    Ok(())
}