fn main() {
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile(
            &[
                "../../proto/risk.proto",
                "../../proto/orders.proto",
                "../../proto/agents.proto",
            ],
            &["../../proto"],
        )
        .expect("Failed to compile proto files");
}
