//! Method pack plugin interface.
//!
//! A method pack is a self-contained crate that registers new mathematics
//! (operators, preconditioners, solvers, problem builders) into Spinoza
//! without modifying core. Packs are discovered at link time via the
//! `inventory` crate.
//!
//! Use function-pointer entry points so that `inventory::submit!` only
//! stores const-constructible values (no `Default::default()` at static
//! init time).

use crate::capability::Capability;
use crate::method_registry::MethodRegistry;

/// A method pack that can register capabilities and factories.
///
/// Implement this trait in your pack crate, then submit an instance
/// via the [`spinoza_register_pack!`] macro.
pub trait MethodPack: Send + Sync + 'static {
    /// Human-readable pack name.
    fn name(&self) -> &'static str;
    /// Pack version string.
    fn version(&self) -> &'static str;
    /// Capabilities this pack provides.
    fn capabilities(&self) -> &'static [Capability];
    /// Register factories into the method registry.
    fn register(&self, registry: &mut MethodRegistry);
}

/// Collected entry stored by `inventory`. Fields are function pointers so
/// the struct is const-constructible.
pub struct PackEntry {
    pub name_fn: fn() -> &'static str,
    pub version_fn: fn() -> &'static str,
    pub capabilities_fn: fn() -> &'static [Capability],
    pub register_fn: fn(&mut MethodRegistry),
}

impl PackEntry {
    pub fn name(&self) -> &'static str {
        (self.name_fn)()
    }
    pub fn version(&self) -> &'static str {
        (self.version_fn)()
    }
    pub fn capabilities(&self) -> &'static [Capability] {
        (self.capabilities_fn)()
    }
    pub fn register(&self, registry: &mut MethodRegistry) {
        (self.register_fn)(registry)
    }
}

// SAFETY: all function pointers are Send + Sync.
unsafe impl Send for PackEntry {}
unsafe impl Sync for PackEntry {}

inventory::collect!(PackEntry);

/// Iterate all registered method packs.
pub fn iter_packs() -> impl Iterator<Item = &'static PackEntry> {
    inventory::iter::<PackEntry>()
}

/// Register a method pack type.
///
/// The macro emits four module-level helper functions and one
/// `inventory::submit!` call. The helpers instantiate the pack via
/// `Default::default()` **at runtime** (when the function is first called),
/// so no const-eval limitation applies.
///
/// Usage:
/// ```ignore
/// #[derive(Default)]
/// struct MyPack;
/// impl MethodPack for MyPack { ... }
/// spinoza_register_pack!(MyPack);
/// ```
#[macro_export]
macro_rules! spinoza_register_pack {
    ($pack_ty:ty) => {
        fn __spinoza_pack_name() -> &'static str {
            <$pack_ty as $crate::MethodPack>::name(
                &<$pack_ty as ::std::default::Default>::default(),
            )
        }
        fn __spinoza_pack_version() -> &'static str {
            <$pack_ty as $crate::MethodPack>::version(
                &<$pack_ty as ::std::default::Default>::default(),
            )
        }
        fn __spinoza_pack_capabilities() -> &'static [$crate::Capability] {
            <$pack_ty as $crate::MethodPack>::capabilities(
                &<$pack_ty as ::std::default::Default>::default(),
            )
        }
        fn __spinoza_pack_register(registry: &mut $crate::MethodRegistry) {
            <$pack_ty as $crate::MethodPack>::register(
                &<$pack_ty as ::std::default::Default>::default(),
                registry,
            )
        }
        $crate::inventory::submit! {
            $crate::PackEntry {
                name_fn: __spinoza_pack_name,
                version_fn: __spinoza_pack_version,
                capabilities_fn: __spinoza_pack_capabilities,
                register_fn: __spinoza_pack_register,
            }
        }
    };
}
