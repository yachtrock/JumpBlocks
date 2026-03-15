use std::path::PathBuf;

fn main() {
    let vendor = PathBuf::from("vendor/src");

    // Core OpenMesh .cc files we need (mesh kernel, connectivity, utils)
    let core_sources = [
        "OpenMesh/Core/Mesh/ArrayKernel.cc",
        "OpenMesh/Core/Mesh/BaseKernel.cc",
        "OpenMesh/Core/Mesh/PolyConnectivity.cc",
        "OpenMesh/Core/Mesh/TriConnectivity.cc",
        "OpenMesh/Core/System/omstream.cc",
        "OpenMesh/Core/Utils/BaseProperty.cc",
        "OpenMesh/Core/Utils/Endian.cc",
        "OpenMesh/Core/Utils/PropertyCreator.cc",
        "OpenMesh/Core/Utils/RandomNumberGenerator.cc",
    ];

    let mut build = cc::Build::new();

    build
        .cpp(true)
        .std("c++17")
        .warnings(false)
        .include(&vendor)
        .include("src")
        .define("OM_STATIC_BUILD", None)
        // Suppress deprecated warnings from OpenMesh internals
        .define("OM_SUPPRESS_DEPRECATED", None);

    // Add OpenMesh core sources
    for src in &core_sources {
        build.file(vendor.join(src));
    }

    // Add our shim
    build.file("src/shim.cpp");

    build.compile("openmesh_shim");

    // Link C++ standard library
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=stdc++");
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=c++");

    println!("cargo:rerun-if-changed=src/shim.cpp");
    println!("cargo:rerun-if-changed=src/shim.h");
}
