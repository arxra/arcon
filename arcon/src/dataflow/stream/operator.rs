use crate::data::ArconType;
use crate::dataflow::stream::Stream;
use crate::dataflow::{
    builder::OperatorBuilder,
    conf::{DefaultBackend, ParallelismStrategy},
    constructor::*,
    dfg::{DFGNode, DFGNodeKind},
};
use crate::stream::operator::Operator;
use std::rc::Rc;
use std::sync::Arc;

use super::keyed::KeyedStream;

/// Extension trait for creating an [Operator]
pub trait OperatorExt<T: ArconType> {
    /// Add an [`Operator`] to the dataflow graph
    ///
    /// Example
    /// ```no_run
    /// use arcon::prelude::*;
    /// let stream: Stream<u64> = Application::default()
    ///     .iterator(0u64..100, |conf| {
    ///         conf.set_arcon_time(ArconTime::Process);
    ///     })
    ///     .operator(OperatorBuilder {
    ///         operator: Arc::new(|| Map::new(|x| x + 10)),
    ///         state: Arc::new(|_| EmptyState),
    ///         conf: Default::default(),
    ///     });
    /// ```
    fn operator<OP: Operator<IN = T> + 'static>(
        self,
        builder: OperatorBuilder<OP>,
    ) -> Stream<OP::OUT>;
}

impl<T: ArconType> OperatorExt<T> for Stream<T> {
    #[must_use]
    fn operator<OP: Operator<IN = T> + 'static>(
        mut self,
        builder: OperatorBuilder<OP>,
    ) -> Stream<OP::OUT> {
        // No more mutations on the previous node, move it from the stream.current_node to the DFG Graph
        self.move_last_node();

        // Set up directory for the operator and create Backend
        let mut state_dir = self.ctx.app.arcon_conf().state_dir();
        let state_id = builder.state_id();
        state_dir.push(state_id);

        let paralellism = match builder.conf.parallelism_strategy {
            ParallelismStrategy::Static(num) => num,
            _ => unreachable!("Managed Parallelism not Supported yet"),
        };

        let prev_dfg_node = self.ctx.app.dfg.get_mut(&self.prev_dfg_id);
        let incoming_channels = prev_dfg_node.get_node_ids();
        let operator_id = prev_dfg_node.get_operator_id() + 1;

        let node_constructor = NodeConstructor::<OP, DefaultBackend>::new(
            format!("Operator_{}", operator_id),
            state_dir,
            Arc::new(builder),
            self.ctx.app.arcon_logger.clone(),
            self.key_builder.take(),
        );

        let dfg_node = DFGNode::new(
            DFGNodeKind::Placeholder, // The NodeFactory will be inserted into the DFG when it is finalized
            operator_id,
            paralellism,
            incoming_channels,
        );
        prev_dfg_node.set_outgoing_channels(dfg_node.get_node_ids());
        let next_dfg_id = self.ctx.app.dfg.insert(dfg_node);

        self.prev_dfg_id = next_dfg_id;
        Stream {
            prev_dfg_id: self.prev_dfg_id,
            ctx: self.ctx,
            last_node: Some(Rc::new(node_constructor)),
            key_builder: None,
            source: None,
        }
    }
}

impl<T: ArconType> OperatorExt<T> for KeyedStream<T> {
    fn operator<OP: Operator<IN = T> + 'static>(
        self,
        builder: OperatorBuilder<OP>,
    ) -> Stream<OP::OUT> {
        self.stream.operator(builder)
    }
}
