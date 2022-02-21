use std::sync::Arc;

use crate::table::MutableTable;
use arrow::{
    array::{Array, BooleanArray, PrimitiveArray, Utf8Array},
    datatypes::{DataType, Schema},
    error::ArrowError,
};

/// Represents an Arcon type that can be converted to Arrow
pub trait ToArrow {
    /// Type to help the runtime know which builder to use
    type Builder: Array;
    /// Returns the underlying Arrow [DataType]
    fn arrow_type() -> DataType;
    /// Return the Arrow Schema
    fn schema() -> Schema;
    /// Creates a new MutableTable
    fn table() -> MutableTable;
    /// Used to append `self` to an Arrow Array
    fn append(
        self,
        builder: &mut Vec<Arc<dyn Array>>,
        timestamp: Option<u64>,
    ) -> Result<(), ArrowError>;
}

macro_rules! to_arrow {
    ($type:ty, $builder_type:ty, $arrow_type:expr) => {
        impl ToArrow for $type {
            type Builder = $builder_type;

            fn arrow_type() -> DataType {
                $arrow_type
            }
            fn schema() -> Schema {
                unreachable!(
                    "Operation not possible for single value {}",
                    stringify!($type)
                );
            }
            fn table() -> MutableTable {
                unreachable!(
                    "Operation not possible for single value {}",
                    stringify!($type)
                );
            }
            fn append(self, _: &mut Vec<Arc<dyn Array>>, _: Option<u64>) -> Result<(), ArrowError> {
                unreachable!(
                    "Operation not possible for single value {}",
                    stringify!($type)
                );
            }
        }
    };
}

// Map types to Arrow Types
to_arrow!(u64, PrimitiveArray<u64>, DataType::UInt64);
to_arrow!(u32, PrimitiveArray<u32>, DataType::UInt32);
to_arrow!(i64, PrimitiveArray<i64>, DataType::Int64);
to_arrow!(i32, PrimitiveArray<i32>, DataType::Int32);
to_arrow!(f64, PrimitiveArray<f64>, DataType::Float64);
to_arrow!(f32, PrimitiveArray<f32>, DataType::Float32);
to_arrow!(bool, BooleanArray, DataType::Boolean);
to_arrow!(String, Utf8Array<i32>, DataType::Utf8);
to_arrow!(Vec<u8>, PrimitiveArray<u8>, DataType::Binary);
