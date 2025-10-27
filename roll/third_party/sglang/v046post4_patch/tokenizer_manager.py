import os
from typing import Optional, Tuple
import fastapi

from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.managers.tokenizer_manager import TokenizerManager, _Communicator

from roll.third_party.sglang.io_struct import (
    SetupCollectiveGroupReqInput,
    BroadcastBucketReqInput,
    BroadcastParameterReqInput,
    UpdateParameterInBucketReqInput,
    UpdateParameterReqInput,
    SetupCollectiveGroupReqOutput,
    BroadcastBucketReqOutput,
    BroadcastParameterReqOutput,
    UpdateParameterInBucketReqOutput,
    UpdateParameterReqOutput,
)

class TokenizerManagerSA(TokenizerManager):
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        super().__init__(server_args=server_args, port_args=port_args)

        self.setup_collective_group_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.broadcast_bucket_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.broadcast_parameter_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.update_parameter_in_bucket_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.update_parameter_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )

        communicator_patch = [(
                    SetupCollectiveGroupReqOutput,
                    self.setup_collective_group_communicator.handle_recv,
                ),
                (
                    BroadcastBucketReqOutput,
                    self.broadcast_bucket_communicator.handle_recv,
                ),
                (
                    BroadcastParameterReqOutput,
                    self.broadcast_parameter_communicator.handle_recv,
                ),
                (
                    UpdateParameterInBucketReqOutput,
                    self.update_parameter_in_bucket_communicator.handle_recv,
                ),
                (
                    UpdateParameterReqOutput,
                    self.update_parameter_communicator.handle_recv,
                )]
        
        self._result_dispatcher._mapping += communicator_patch
    
    async def setup_collective_group(
        self,
        obj: SetupCollectiveGroupReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1
        ), "dp_size must be 1 for init parameter update group"
        result = (await self.setup_collective_group_communicator(obj))[0]
        return result.success, result.message

    async def broadcast_bucket(
        self,
        obj: BroadcastBucketReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1
        ), "dp_size must be 1 for init parameter update group"
        result = (await self.broadcast_bucket_communicator(obj))[0]
        return result.success, result.message

    async def broadcast_parameter(
        self,
        obj: BroadcastParameterReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1
        ), "dp_size must be 1 for init parameter update group"
        result = (await self.broadcast_parameter_communicator(obj))[0]
        return result.success, result.message

    async def update_parameter(
        self,
        obj: UpdateParameterReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1
        ), "dp_size must be 1 for init parameter update group"
        result = (await self.update_parameter_communicator(obj))[0]
        return result.success, result.message

    async def update_parameter_in_bucket(
        self,
        obj: UpdateParameterInBucketReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1
        ), "dp_size must be 1 for init parameter update group"
        result = (await self.update_parameter_in_bucket_communicator(obj))[0]
        return result.success, result.message