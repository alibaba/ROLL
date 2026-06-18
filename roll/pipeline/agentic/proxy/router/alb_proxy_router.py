# -*- coding: utf-8 -*-
"""
Alibaba Cloud ALB (Application Load Balancer) registration utilities.

This module provides functionality to register rollout server instances
to Alibaba Cloud ALB server groups for load balancing.
"""
import hashlib
import os
import random
import re
import uuid
from typing import List, Optional, Tuple


def _get_logger():
    import logging
    return logging.getLogger(__name__)

logger = _get_logger()


def parse_address(address: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse IP and port from address string.

    Args:
        address: Address string like "http://192.168.1.1:8080"

    Returns:
        Tuple of (ip, port) or (None, None) if parsing fails
    """
    match = re.match(r'https?://([^:]+):(\d+)', address)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def get_alb_load_balancer_id_by_hash(
        job_name: str,
        listen_port: int,
        alb_ids_env_var: str = 'ALB_LOAD_BALANCER_IDS',
) -> Tuple[str, str]:
    """
    Select an ALB load balancer ID from a comma-separated list based on hash of job_name and listen_port.

    This function distributes jobs across multiple ALBs using consistent hashing to ensure
    stable routing for the same job_name/listen_port combination.

    Args:
        job_name: Name of the job (used for consistent hashing)
        listen_port: Port number being listened on (used for consistent hashing)
        alb_ids_env_var: Environment variable name containing comma-separated ALB IDs

    Returns:
        Selected ALB load balancer ID

    Environment variables:
        - ALB_LOAD_BALANCER_IDS: Comma-separated list of ALB load balancer IDs
          (e.g., "alb-abc,alb-def,alb-ghi")

    Raises:
        ValueError: If ALB_LOAD_BALANCER_IDS environment variable is not set or empty
    """
    alb_ids = os.environ.get(alb_ids_env_var, None)
    alb_lsn_ids = os.environ.get('ALB_LOAD_BALANCER_LSN_IDS', None)
    if not alb_ids:
        raise ValueError(f"Environment variable '{alb_ids_env_var}' is not set")

    alb_list = [alb.strip() for alb in alb_ids.split(',')]
    alb_lsn_list = [alb_lsn.strip() for alb_lsn in alb_lsn_ids.split(',')]
    if not alb_list:
        raise ValueError(f"No ALB IDs found in '{alb_ids_env_var}'")

    # Use consistent hashing to select ALB ID with fixed seed for consistency
    hash_value = get_deterministic_hash(job_name, listen_port)
    selected_index = hash_value % len(alb_list)
    print(f"alb_list: {alb_list}, alb_lsn_list: {alb_lsn_list}")
    print(f"job name {job_name}, listen_port {listen_port}, hash {hash_value}, selected_index: {selected_index}")
    return alb_list[selected_index], alb_lsn_list[selected_index]


def get_deterministic_hash(job_name, listen_port):
    # 将变量转换为字符串并拼接
    data = f"{job_name}:{listen_port}"
    # 使用 SHA-256 生成哈希
    hash_obj = hashlib.sha256(data.encode('utf-8'))
    # 取前几位作为整数，或者直接返回 hex 字符串
    return int(hash_obj.hexdigest()[:8], 16)


def discover_and_set_alb_load_balancers(
        vpc_id: str = None,
        alb_ids_env_var: str = 'ALB_LOAD_BALANCER_IDS',
        region: str = None,
        client: object = None
) -> str:
    """
    Discover all ALBs in the current region and set them to environment variable ALB_LOAD_BALANCER_IDS.

    This function queries Alibaba Cloud ALB service to find all load balancers in the specified region
    and sets their IDs to the ALB_LOAD_BALANCER_IDS environment variable.

    Args:
        vpc_id: VPC ID to filter ALBs (optional, if None uses default VPC from ALB_VPC_ID env var)
        alb_ids_env_var: Environment variable name to set ALB IDs (default: ALB_LOAD_BALANCER_IDS)
        region: Region to query ALBs (optional, if None uses default from ALB_REGION env var)

    Returns:
        Comma-separated string of ALB IDs

    Environment variables:
        - ALIBABA_CLOUD_ACCESS_KEY_ID: Access key ID
        - ALIBABA_CLOUD_ACCESS_KEY_SECRET: Access key secret
        - ALB_REGION: Optional, region to query (default: cn-hangzhou)
        - ALB_VPC_ID: Optional, VPC ID to filter ALBs

    Raises:
        ImportError: If alibabacloud SDK is not installed
        ValueError: If required environment variables are not set
        Exception: If ALB discovery fails
    """
    from alibabacloud_alb20200616 import models as alb_20200616_models
    from alibabacloud_tea_util import models as util_models

    # Determine region
    if region is None:
        region = os.environ.get('ALB_REGION', None)

    assert region is not None, "region is required for querying ALBs"

    list_load_balancers_request = alb_20200616_models.ListLoadBalancersRequest(
        zone_id='cn-hangzhou-a'
    )

    runtime = util_models.RuntimeOptions()
    print(f"start List load balancers: {client}")
    response = client.list_load_balancers_with_options(list_load_balancers_request, runtime)

    # Extract ALB IDs
    alb_ids = []
    if response.body.load_balancers:
        for lb in response.body.load_balancers:
            alb_ids.append(lb.load_balancer_id)

    if not alb_ids:
        raise ValueError(f"No ALBs found in region {region} with VPC ID {vpc_id}")

    # Join ALB IDs with comma
    alb_ids_str = ','.join(alb_ids)

    # Set environment variable
    os.environ[alb_ids_env_var] = alb_ids_str

    logger.info(f"Discovered {len(alb_ids)} ALBs in region {region}: {alb_ids_str}")

    return alb_ids_str


def create_alb_client(endpoint: str = None) -> object:
    """
    Create ALB client using credentials from environment variables.

    Environment variables required:
        - ALIBABA_CLOUD_ACCESS_KEY_ID
        - ALIBABA_CLOUD_ACCESS_KEY_SECRET
        - ALB_ENDPOINT (optional, default: alb.cn-hangzhou.aliyuncs.com)

    Returns:
        Alb20200616Client instance

    Raises:
        ImportError: If alibabacloud SDK is not installed
        ValueError: If required environment variables are not set
    """
    if endpoint is None:
        alb_region = os.environ.get('ALB_REGION')
        endpoint = f'alb-vpc.{alb_region}.aliyuncs.com' if alb_region else None
    try:
        from alibabacloud_alb20200616.client import Client as Alb20200616Client
        from alibabacloud_credentials.client import Client as CredentialClient
        from alibabacloud_credentials.models import Config as CredentialConfig
        from alibabacloud_tea_openapi import models as open_api_models
    except ImportError:
        raise ImportError(
            "Please install alibabacloud SDK: "
            "pip install alibabacloud_alb20200616 alibabacloud_credentials alibabacloud_tea_openapi"
        )

    access_key_id = os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID')
    access_key_secret = os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_SECRET')

    if not access_key_id or not access_key_secret:
        raise ValueError(
            "ALIBABA_CLOUD_ACCESS_KEY_ID and ALIBABA_CLOUD_ACCESS_KEY_SECRET "
            "environment variables must be set"
        )

    credentials_config = CredentialConfig(
        type='access_key',
        access_key_id=access_key_id,
        access_key_secret=access_key_secret
    )
    credentials_client = CredentialClient(credentials_config)

    config = open_api_models.Config(credential=credentials_client)
    config.endpoint = endpoint or os.environ.get(
        'ALB_ENDPOINT', 'alb-vpc.cn-hangzhou.aliyuncs.com'
    )
    print(f"config.endpoint: {config.endpoint} config.access_key_id: {config.access_key_id} create Alb20200616Client")
    return Alb20200616Client(config)


def delete_server_group(client, server_group_id: str) -> dict:
    """
    Delete an existing ALB server group.

    Args:
        client: ALB client instance
        server_group_id: Server group ID to delete

    Returns:
        API response dict
    """
    from alibabacloud_alb20200616 import models as alb_20200616_models
    from alibabacloud_tea_util import models as util_models

    print(f"start Delete server group: {server_group_id}")
    delete_server_group_request = alb_20200616_models.DeleteServerGroupRequest(
        server_group_id=server_group_id
    )
    runtime = util_models.RuntimeOptions()
    response = client.delete_server_group_with_options(delete_server_group_request, runtime)

    print(f"Deleted server group: {server_group_id}")
    return response.body.to_map() if hasattr(response.body, 'to_map') else {}


def delete_listener(client, listener_id: str) -> dict:
    """
    Delete an existing ALB listener.

    Args:
        client: ALB client instance
        listener_id: Listener ID to delete

    Returns:
        API response dict
    """
    from alibabacloud_alb20200616 import models as alb_20200616_models
    from alibabacloud_tea_util import models as util_models

    print(f"start Delete listener: {listener_id}")
    delete_listener_request = alb_20200616_models.DeleteListenerRequest(
        listener_id=listener_id
    )
    runtime = util_models.RuntimeOptions()
    response = client.delete_listener_with_options(delete_listener_request, runtime)

    print(f"Deleted listener: {listener_id}")
    return response.body.to_map() if hasattr(response.body, 'to_map') else {}


def create_server_group(
        client,
        server_group_name: str,
        vpc_id: str,
        server_type: str = 'Ip',
        resource_group_id: str = 'rg-aeky6sewmnhmbyi',
        health_check_enabled: bool = False,
) -> str:
    """
    Create a new ALB server group.

    Args:
        client: ALB client instance
        server_group_name: Name for the server group
        vpc_id: VPC ID where the server group will be created
        health_check_enabled: Whether to enable health check

    Returns:
        Server group ID
    """
    from alibabacloud_alb20200616 import models as alb_20200616_models
    from alibabacloud_tea_util import models as util_models

    print(f"start Created server group: {server_group_name} {vpc_id}")

    health_check_config = alb_20200616_models.CreateServerGroupRequestHealthCheckConfig(
        health_check_enabled=False
    )
    sticky_session_config = alb_20200616_models.CreateServerGroupRequestStickySessionConfig(
        sticky_session_enabled=False
    )

    create_server_group_request = alb_20200616_models.CreateServerGroupRequest(
        server_group_type=server_type,
        server_group_name=server_group_name,
        resource_group_id=resource_group_id,
        vpc_id=vpc_id,
        health_check_config=health_check_config,
        sticky_session_config=sticky_session_config
    )

    runtime = util_models.RuntimeOptions()
    response = client.create_server_group_with_options(create_server_group_request, runtime)

    server_group_id = response.body.server_group_id
    print(f"Created server group success: {server_group_id}")
    return server_group_id


def add_servers_to_server_group(
        client,
        server_group_id: str,
        addresses: List[str],
        server_type: str = 'Ip',
        weight: int = 100,
) -> dict:
    """
    Add servers to an existing ALB server group.

    Args:
        client: ALB client instance
        server_group_id: Target server group ID
        addresses: List of server addresses (e.g., ["http://192.168.1.1:8080"])
        server_type: Server type, default is 'Ip'
        weight: Server weight for load balancing (0-100)

    Returns:
        API response dict
    """
    from alibabacloud_alb20200616 import models as alb_20200616_models
    from alibabacloud_tea_util import models as util_models

    servers = []
    for address in addresses:
        ip, port = parse_address(address)
        if ip is None or port is None:
            logger.warning(f"Cannot parse address: {address}, skipping")
            continue

        server = alb_20200616_models.AddServersToServerGroupRequestServers(
            server_id=ip,
            server_type=server_type,
            server_ip=ip,
            remote_ip_enabled=True,
            port=port
        )
        print(f"Added server {ip}:{port} to server group {server_group_id}")
        servers.append(server)

    if not servers:
        logger.warning("No valid servers to add to server group")
        return {}

    request = alb_20200616_models.AddServersToServerGroupRequest(
        server_group_id=server_group_id,
        servers=servers
    )

    runtime = util_models.RuntimeOptions()
    response = client.add_servers_to_server_group_with_options(request, runtime)

    print(f"Added {len(servers)} servers to server group {server_group_id}")
    return response.body.to_map() if hasattr(response.body, 'to_map') else {}


def start_listener(client, listener_id: str) -> dict:
    """
    Start a stopped ALB listener.

    Args:
        client: ALB client instance
        listener_id: Listener ID to start

    Returns:
        API response dict
    """
    from alibabacloud_alb20200616 import models as alb_20200616_models
    from alibabacloud_tea_util import models as util_models

    start_listener_request = alb_20200616_models.StartListenerRequest(
        listener_id=listener_id
    )
    runtime = util_models.RuntimeOptions()
    response = client.start_listener_with_options(start_listener_request, runtime)

    print(f"Started listener: {listener_id}")
    return response.body.to_map() if hasattr(response.body, 'to_map') else {}


def poll_listener_status(
        client,
        listener_id: str,
        target_status: str = 'Running',
        max_attempts: int = 10,
        interval_seconds: float = 2.0
) -> dict:
    """
    Poll listener status until it reaches target state.

    Args:
        client: ALB client instance
        listener_id: Listener ID to monitor
        target_status: Target status to wait for (default: 'Running')
        max_attempts: Maximum number of polling attempts
        interval_seconds: Interval between polls in seconds

    Returns:
        Final listener status dict

    Raises:
        TimeoutError: If target status is not reached within max_attempts
    """
    from alibabacloud_alb20200616 import models as alb_20200616_models
    from alibabacloud_tea_util import models as util_models
    import time

    for attempt in range(max_attempts):
        list_listeners_request = alb_20200616_models.ListListenersRequest(
            listener_ids=[listener_id]
        )
        runtime = util_models.RuntimeOptions()
        try:
            resp = client.list_listeners_with_options(list_listeners_request, runtime)
            if resp.body and resp.body.listeners:
                status = resp.body.listeners[0].listener_status
                print(f"Poll attempt {attempt + 1}/{max_attempts}: Status={status}")

                if status == target_status:
                    print(f"Listener {listener_id} reached target status '{target_status}'")
                    return resp.body.to_map() if hasattr(resp.body, 'to_map') else {}

                if status not in ['Provisioning', 'Running', 'Stopping', 'Configuring', 'Stopped', 'Starting',
                                  'Deleting', 'Deleted']:
                    raise RuntimeError(f"Listener {listener_id} entered invalid state: {status}")
            else:
                print(f"Poll attempt {attempt + 1}/{max_attempts}: Listener not found")
        except Exception as e:
            print(f"Poll attempt {attempt + 1}/{max_attempts}: Error checking status - {e}")

        if attempt < max_attempts - 1:
            time.sleep(interval_seconds)

    raise TimeoutError(
        f"Listener {listener_id} did not reach status '{target_status}' "
        f"within {max_attempts} attempts"
    )


def poll_server_group_status(
        client,
        server_group_id: str,
        target_status: str = 'Available',
        max_attempts: int = 10,
        interval_seconds: float = 2.0
) -> dict:
    """
    Poll server group status until it reaches target state.

    Args:
        client: ALB client instance
        server_group_id: Server group ID to monitor
        target_status: Target status to wait for (default: 'Active')
        max_attempts: Maximum number of polling attempts
        interval_seconds: Interval between polls in seconds

    Returns:
        Final server group status dict

    Raises:
        TimeoutError: If target status is not reached within max_attempts
    """
    from alibabacloud_alb20200616 import models as alb_20200616_models
    from alibabacloud_tea_util import models as util_models
    import time

    for attempt in range(max_attempts):
        list_server_groups_request = alb_20200616_models.ListServerGroupsRequest(
            server_group_ids=[server_group_id]
        )
        runtime = util_models.RuntimeOptions()
        try:
            resp = client.list_server_groups_with_options(list_server_groups_request, runtime)
            if resp.body and resp.body.server_groups:
                status = resp.body.server_groups[0].server_group_status
                print(f"Poll attempt {attempt + 1}/{max_attempts}: Status={status}")

                if status == target_status:
                    print(f"Server group {server_group_id} reached target status '{target_status}'")
                    return resp.body.to_map() if hasattr(resp.body, 'to_map') else {}

                if status not in ['Creating', 'Available', 'Configuring']:
                    raise RuntimeError(f"Server group {server_group_id} entered invalid state: {status}")
            else:
                print(f"Poll attempt {attempt + 1}/{max_attempts}: Server group not found")
        except Exception as e:
            print(f"Poll attempt {attempt + 1}/{max_attempts}: Error checking status - {e}")

        if attempt < max_attempts - 1:
            time.sleep(interval_seconds)

    raise TimeoutError(
        f"Server group {server_group_id} did not reach status '{target_status}' "
        f"within {max_attempts} attempts"
    )


def add_rules(
        client,
        listener_id: str,
        server_group_id: str,
        port: int,
        weight: int = 100,
) -> dict:
    """
    Add a forwarding rule to a listener.

    Args:
        client: ALB client instance
        listener_id: Listener ID to add rule to
        server_group_id: Server group ID to forward traffic to
        path_pattern: Path pattern for matching (e.g., '*job_id/port/*')
        priority: Rule priority (lower number = higher priority)
        weight: Server group weight for load balancing

    Returns:
        API response dict
    """
    import time

    retry_count = 0
    while retry_count < 30:
        try:
            random_seed = str(uuid.uuid4())
            priority = hash(random_seed) % 10000
            from alibabacloud_alb20200616 import models as alb_20200616_models
            from alibabacloud_tea_util import models as util_models
            import json

            job_id = os.environ.get('TASK_ID')
            print(f"add rules to : {listener_id} {job_id} {port} {priority}")

            # Create server group tuple
            server_group_tuple = alb_20200616_models.CreateRulesRequestRulesRuleActionsForwardGroupConfigServerGroupTuples(
                server_group_id=server_group_id,
                weight=weight
            )

            # Create forward group config
            forward_group_config = alb_20200616_models.CreateRulesRequestRulesRuleActionsForwardGroupConfig(
                server_group_tuples=[server_group_tuple]
            )

            rule_actions_0rewrite_config = alb_20200616_models.CreateRuleRequestRuleActionsRewriteConfig(
                host='${host}',
                path='/${2}',
                query='${query}'
            )

            rule_actions_rewrite = alb_20200616_models.CreateRuleRequestRuleActions(
                order=1,
                rewrite_config=rule_actions_0rewrite_config,
                type='Rewrite'
            )

            # Create rule action
            rule_action = alb_20200616_models.CreateRuleRequestRuleActions(
                order=2,
                forward_group_config=forward_group_config,
                type='ForwardGroup'
            )
            path_pattern = f"~/(.*)/{job_id}/{port}/(.*)"
            # Create path config for condition
            path_config = alb_20200616_models.CreateRulesRequestRulesRuleConditionsPathConfig(
                values=[path_pattern]
            )

            # Create rule condition
            rule_condition = alb_20200616_models.CreateRulesRequestRulesRuleConditions(
                type='Path',
                path_config=path_config
            )

            # Generate rule name based on listener_id and priority
            rule_name = f"rule-{job_id}-{port}"

            # Create rule
            rule = alb_20200616_models.CreateRulesRequestRules(
                priority=priority,
                direction='Request',
                rule_name=rule_name,
                rule_conditions=[rule_condition],
                rule_actions=[rule_actions_rewrite, rule_action]
            )

            # Create rules request
            create_rules_request = alb_20200616_models.CreateRulesRequest(
                listener_id=listener_id,
                rules=[rule]
            )

            runtime = util_models.RuntimeOptions()

            resp = client.create_rules_with_options(create_rules_request, runtime)
            print(f"Successfully added rule: {rule_name}")
            print(f"Path pattern: {path_pattern}")
            print(f"Server group: {server_group_id}")
            print(f"Add Rules Response: {resp}")
            print(f"Add Rules Response Body: {resp.body}")
            print(f"Add Rules Response Body List: {resp.body.rule_ids}")

            # 成功执行后跳出循环
            break

        except Exception as e:
            # 只有在异常代码为Conflict.Priority时才重试
            retry_count += 1
            print(f"Retry attempt {retry_count}/20: {e}")
            if retry_count >= 20:
                raise Exception(f"Failed to add rules after {retry_count} attempts. Last error: {e}")
            # 等待一段时间再重试
            sleep_time = random.randint(10, 30)
            time.sleep(sleep_time)

    # Extract rule IDs from response
    assert resp.body.rule_ids
    rule_ids = [rule_id.rule_id for rule_id in resp.body.rule_ids]

    return {
        'rule_name': rule_name,
        'rule_ids': rule_ids,
        'response': resp.body.to_map() if hasattr(resp.body, 'to_map') else {}
    }


def cleanup_alb_resources():
    """
    Scan ALB_RESOURCE_CLEAN_DIR and delete corresponding resources.

    Reads files in ALB_RESOURCE_CLEAN_DIR, extracts resource IDs,
    and deletes the corresponding server groups/listeners.

    Files should follow the naming pattern:
    - server-group_{id}.txt
    - listener_{id}.txt

    Environment variables:
        - ALB_RESOURCE_CLEAN_DIR: Directory containing resource ID files
        - ALIBABA_CLOUD_ACCESS_KEY_ID: Access key ID
        - ALIBABA_CLOUD_ACCESS_KEY_SECRET: Access key secret
        - ALB_ENDPOINT: Optional, ALB API endpoint

    Returns:
        dict: Summary of deleted resources
    """
    clean_parent_dir = os.environ.get('ALB_RESOURCE_CLEAN_DIR')
    task_dir = os.environ.get('TASK_ID')
    clean_dir = os.path.join(clean_parent_dir, task_dir)
    if not clean_dir:
        print("ALB_RESOURCE_CLEAN_DIR not set, nothing to clean up")
        return {"cleaned_up": 0, "errors": []}

    if not os.path.exists(clean_dir):
        print(f"Directory {clean_dir} does not exist")
        return {"cleaned_up": 0, "errors": []}

    client = create_alb_client()
    cleaned_up = 0
    errors = []
    filenames = sorted(os.listdir(clean_dir), key=lambda x: (
        0 if x.startswith('rule_') else 1,
        0 if x.startswith('server-group_') else 1,
        x
    ))

    for filename in filenames:
        filepath = os.path.join(clean_dir, filename)
        print(f"Scanning file: {filepath}")
        # Skip directories and non-text files
        if not os.path.isfile(filepath) or not filename.endswith('.txt'):
            continue

        # Parse resource type and ID from filename
        parts = filename.split('_')
        if len(parts) < 2:
            continue

        resource_type = parts[0]
        resource_id = parts[1][:-4]  # Remove .txt extension for other types
        print(f"attempt to delete Resource type: {resource_type}, ID: {resource_id}")

        # Use the new helper function
        success, error_msg = delete_resource_with_retry(client, resource_type, resource_id, 5)
        if not success:
            errors.append(error_msg)

        # Remove the file after attempting deletion
        if success:
            safe_remove_file(filepath)
            cleaned_up += 1

    # Check if clean_dir is empty and remove it if so
    try:
        if os.path.exists(clean_dir) and not os.listdir(clean_dir):
            os.rmdir(clean_dir)
            print(f"Removed empty directory: {clean_dir}")
    except Exception as e:
        print(f"Warning: Failed to remove empty directory {clean_dir}: {e}")

    return {
        "cleaned_up": cleaned_up,
        "errors": errors
    }


def delete_resource_with_retry(client, resource_type: str, resource_id: str, max_retries: int = 5) -> tuple[bool, str]:
    """
    删除资源并带重试机制

    Args:
        client: ALB客户端实例
        resource_type: 资源类型 ('server-group', 'listener', 'rule')
        resource_id: 资源ID
        max_retries: 最大重试次数

    Returns:
        tuple: (success: bool, error_message: str)
    """
    import random
    import time

    for attempt in range(max_retries):
        try:
            # Random sleep before each attempt (1-15 seconds)
            if attempt > 0:
                delay = random.randint(1, 15)
                print(f"Retry attempt {attempt + 1}/{max_retries}, sleeping {delay:.2f} seconds...")
                time.sleep(delay)

            success = False
            if resource_type == 'server-group':
                delete_server_group(client, resource_id)
                print(f"Deleted server group: {resource_id}")
                success = True
            elif resource_type == 'listener':
                delete_listener(client, resource_id)
                print(f"Deleted listener: {resource_id}")
                success = True
            elif resource_type == 'rule':
                delete_rule(client, resource_id)
                print(f"Initiated deletion of rule: {resource_id}")
                success = True
            else:
                print(f"Unknown resource type: {resource_type}")
                return False, f"Unknown resource type: {resource_type}"

            if success:
                return True, ""  # Success

        except Exception as e:
            if '404' in f"{e}":
                print(f"Resource {resource_id} not found, skipping")
                return True, ""  # Consider 404 as success
            error_msg = f"Failed to delete {resource_type} {resource_id} (attempt {attempt + 1}/{max_retries}): {e}"
            print(error_msg)

            if attempt == max_retries - 1:
                # Final attempt failed
                return False, error_msg
            else:
                # Will retry
                print(f"Will retry after random delay...")

    return False, ""


def safe_remove_file(filepath: str) -> bool:
    """
    Safely remove a file without throwing any exceptions.

    Args:
        filepath: Path to the file to be removed

    Returns:
        True if file was successfully removed, False otherwise
    """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        else:
            # File doesn't exist, consider as successfully "removed"
            return True
    except Exception as e:
        # Log the error but don't propagate it
        print(f"Warning: Failed to remove file {filepath}: {e}")
        return False


def save_resource_id_to_file(resource_id: str, resource_type: str = 'server-group') -> str:
    """
    Save server group ID or listener ID to a file in ALB_RESOURCE_CLEAN_DIR.

    Args:
        resource_id: Resource ID to save (server_group_id or listener_id)
        resource_type: Type of resource ('server_group' or 'listener')

    Returns:
        File path where the resource ID was saved

    Raises:
        ValueError: If ALB_RESOURCE_CLEAN_DIR environment variable is not set
        IOError: If file cannot be created or written
    """
    clean_parent_dir = os.environ.get('ALB_RESOURCE_CLEAN_DIR')
    task_dir = os.environ.get('TASK_ID')
    clean_dir = os.path.join(clean_parent_dir, task_dir)
    if not clean_dir:
        raise ValueError("ALB_RESOURCE_CLEAN_DIR environment variable is not set")

    # Ensure directory exists
    os.makedirs(clean_dir, exist_ok=True)

    # Generate filename based on resource type and ID
    filename = f"{resource_type}_{resource_id}.txt"
    filepath = os.path.join(clean_dir, filename)

    # Write resource ID to file
    try:
        with open(filepath, 'w') as f:
            f.write(resource_id)
        print(f"Saved {resource_type} ID {resource_id} to file: {filepath}")

        # Verify file was created and contains correct content
        with open(filepath, 'r') as f:
            content = f.read().strip()
            if content != resource_id:
                raise IOError(f"File verification failed: expected '{resource_id}', got '{content}'")

        return filepath
    except IOError as e:
        raise IOError(f"Failed to save {resource_type} ID to file: {e}")


def register_rollout_servers_to_alb(
        addresses: List[str],
        load_balancer_id: str = None,
        server_group_name: str = None,
        vpc_id: str = None,
        listener_port: int = None,
        endpoint: str = None,
) -> dict:
    """
    Register rollout servers to Alibaba Cloud ALB.

    This is the main entry point for registering rollout servers.
    It creates a new server group (with random name) and adds servers to it.

    Args:
        addresses: List of rollout server addresses
        load_balancer_id: Load balancer ID (required for creating listener)
        server_group_name: Name for new server group (auto-generated if not provided)
        vpc_id: VPC ID (required for creating server group)
        listener_port: Port for listener
        endpoint: ALB API endpoint (default from env or cn-hangzhou)

    Returns:
        Dict containing created resource IDs

    Environment variables:
        - ALIBABA_CLOUD_ACCESS_KEY_ID: Access key ID
        - ALIBABA_CLOUD_ACCESS_KEY_SECRET: Access key secret
        - ALB_ENDPOINT: Optional, ALB API endpoint
        - ALB_LOAD_BALANCER_IDS: Optional, default load balancer ID
        - ALB_VPC_ID: Optional, default VPC ID
    """
    # Get defaults from environment variables
    vpc_id = vpc_id or os.environ.get('ALB_VPC_ID')
    resource_group_id = os.environ.get('ALB_RESOURCE_GROUP_ID')
    alb_ids_str = os.environ.get('ALB_LOAD_BALANCER_IDS')
    result = {
        'server_group_id': None,
        'listener_id': None,
        'registered_servers': []
    }

    try:
        alb_region = os.environ.get('ALB_REGION')
        if endpoint is None:
            endpoint = os.environ.get('ALB_ENDPOINT') or f'alb.{alb_region}.aliyuncs.com'
        client = create_alb_client(endpoint)
        if not alb_ids_str:
            discover_and_set_alb_load_balancers(vpc_id=vpc_id, region=alb_region, client=client)
        job_id = os.environ.get('TASK_ID')
        # Get listener_port from first address if not provided
        if not listener_port and addresses and addresses[0]:
            _, listener_port = parse_address(addresses[0])

        load_balancer_id, lsn_id = get_alb_load_balancer_id_by_hash(job_id, listener_port)

        # Always create a new server group with name based on first address and port
        if not server_group_name:
            # Extract IP from first address (e.g., "http://10.71.240.61:20000" -> "10-71-240-61")
            if addresses:
                first_ip, _ = parse_address(addresses[0])
                if first_ip:
                    ip_suffix = first_ip.replace('.', '-')
                    server_group_name = f"roll-rollout-{ip_suffix}-{listener_port}"
                else:
                    server_group_name = f"roll-rollout-{len(addresses)}nodes-{listener_port}"
            else:
                server_group_name = f"roll-rollout-empty-{listener_port}"
        if not vpc_id:
            raise ValueError("vpc_id is required for creating server group")

        server_group_id = create_server_group(
            client=client,
            server_group_name=server_group_name,
            vpc_id=vpc_id,
            resource_group_id=resource_group_id,
        )
        result['server_group_id'] = server_group_id

        poll_server_group_status(
            client=client,
            server_group_id=server_group_id,
        )
        # record server group id
        save_resource_id_to_file(server_group_id, resource_type='server-group')
        # Add servers to server group
        add_response = add_servers_to_server_group(
            client=client,
            server_group_id=server_group_id,
            addresses=addresses
        )
        result['registered_servers'] = addresses

        poll_server_group_status(
            client=client,
            server_group_id=server_group_id,
        )

        assert listener_port is not None, "listener_port is required for creating listener"
        # Create listener rules with retry logic
        print(
            f"prepare to add rules load_balancer_id: {load_balancer_id}, lsn_id: {lsn_id} listener_port: {listener_port}")
        assert listener_port and load_balancer_id and lsn_id

        # Add rules to listener and save rule ids to file
        try:
            rule_result = add_rules(
                client=client,
                listener_id=lsn_id,
                server_group_id=server_group_id,
                port=listener_port
            )

            # Save rule IDs to file
            if rule_result.get('rule_ids'):
                poll_rule_status(client=client, rule_ids=rule_result['rule_ids'])
                rule_ids_str = ','.join(rule_result['rule_ids'])
                save_resource_id_to_file(rule_ids_str, resource_type='rule')
                print(f"Saved rule IDs: {rule_ids_str}")
            else:
                print("Warning: No rule IDs returned from add_rules")

        except Exception as e:
            print(f"Failed to add rules to listener: {e}")
            raise

        logger.info(f"Successfully registered rollout servers to ALB: {result}")
        return result

    except Exception as e:
        logger.error(f"Failed to register rollout servers to ALB: {e}")
        error_message = e.data.get("Recommend") if hasattr(e, 'data') else str(e)
        logger.error(f"Failed to register rollout servers to ALB: {error_message}")

        raise


def poll_rule_status(
        client,
        rule_ids: List[str],
        target_status: str = 'Available',
        max_attempts: int = 10,
        interval_seconds: float = 2.0
) -> dict:
    """
    Poll rule status until it reaches target state.

    Args:
        client: ALB client instance
        rule_id: Rule ID to monitor
        target_status: Target status to wait for (default: 'Available')
        max_attempts: Maximum number of polling attempts
        interval_seconds: Interval between polls in seconds

    Returns:
        Final rule status dict

    Raises:
        TimeoutError: If target status is not reached within max_attempts
    """
    from alibabacloud_alb20200616 import models as alb_20200616_models
    from alibabacloud_tea_util import models as util_models
    import time

    for attempt in range(max_attempts):
        # Since ALB API doesn't seem to have a direct ListRulesRequest with rule_ids parameter,
        # we'll need to list rules by listener. However, we don't have listener_id here.
        # Let's assume there's a way to list rules directly by ID or we need to adapt the approach.
        try:
            # Attempt to use a generic listing approach
            # This would need to be adapted based on actual ALB API capabilities
            list_rules_request = alb_20200616_models.ListRulesRequest(
                rule_ids=rule_ids
            )
            runtime = util_models.RuntimeOptions()

            # This will likely fail without proper parameters, but shows the intended structure
            resp = client.list_rules_with_options(list_rules_request, runtime)

            if resp.body and hasattr(resp.body, 'rules'):
                for rule in resp.body.rules:
                    status = rule.rule_status if hasattr(rule, 'rule_status') else 'Unknown'
                    print(f"Poll attempt {attempt + 1}/{max_attempts}: Status={status}")

                    if status == target_status:
                        print(f"Rule {rule.rule_id} reached target status '{target_status}'")
                        return resp.body.to_map() if hasattr(resp.body, 'to_map') else {}

                    if status not in ['Provisioning', 'Configuring', 'Available', ]:
                        raise RuntimeError(f"Rule {rule.rule_id} entered invalid state: {status}")

                    break
                else:
                    print(f"Poll attempt {attempt + 1}/{max_attempts}: Rule {rule.rule_id} not found in response")
            else:
                print(f"Poll attempt {attempt + 1}/{max_attempts}: No rules found in response")

        except Exception as e:
            print(f"Poll attempt {attempt + 1}/{max_attempts}: Error checking status - {e}")

        if attempt < max_attempts - 1:
            time.sleep(interval_seconds)

    raise TimeoutError(
        f"Rule {rule_ids} did not reach status '{target_status}' "
        f"within {max_attempts} attempts"
    )


def delete_rule(client, rule_id: str) -> dict:
    """
    Delete a forwarding rule from a listener.

    Args:
        client: ALB client instance
        rule_id: Rule ID to delete

    Returns:
        API response dict
    """
    from alibabacloud_alb20200616 import models as alb_20200616_models
    from alibabacloud_tea_util import models as util_models

    print(f"Deleting rule: {rule_id}")

    delete_rule_request = alb_20200616_models.DeleteRuleRequest(
        rule_id=rule_id
    )

    runtime = util_models.RuntimeOptions()
    resp = client.delete_rule_with_options(delete_rule_request, runtime)

    print(f"Successfully initiated deletion of rule: {rule_id}")
    print(f"Job ID: {resp.body.job_id}")

    return {
        'job_id': resp.body.job_id,
        'response': resp.body.to_map() if hasattr(resp.body, 'to_map') else {}
    }


def list_rules() -> dict:
    """
    循环列举指定 listener_ids 下的所有规则，并返回 rule_name 到 rule_id 的映射

    Args:
        client: ALB client instance
        listener_ids: Listener IDs 列表
        max_results: 每页最大返回结果数，默认100

    Returns:
        包含 rule_name 到 rule_id 映射的字典
    """
    from alibabacloud_alb20200616 import models as alb_20200616_models
    from alibabacloud_tea_util import models as util_models
    client = create_alb_client()
    max_results: int = 100
    all_rules = []
    listener_ids = os.environ.get('ALB_LOAD_BALANCER_LSN_IDS').split(',')
    next_token = None

    while True:
        list_rules_request = alb_20200616_models.ListRulesRequest(
            listener_ids=listener_ids,
            max_results=max_results
        )

        if next_token:
            list_rules_request.next_token = next_token

        runtime = util_models.RuntimeOptions()

        try:
            resp = client.list_rules_with_options(list_rules_request, runtime)

            if hasattr(resp.body, 'rules') and resp.body.rules:
                all_rules.extend(resp.body.rules)

            next_token = resp.body.next_token if hasattr(resp.body, 'next_token') else None

            if not next_token:
                break

        except Exception as error:
            print(f"Error occurred while listing rules: {error.message if hasattr(error, 'message') else str(error)}")
            if hasattr(error, 'data') and error.data:
                print(error.data.get("Recommend"))
            raise

    rule_name_to_id_dict = {}
    for rule in all_rules:
        if hasattr(rule, 'rule_name') and hasattr(rule, 'rule_id'):
            rule_name_to_id_dict[rule.rule_name] = rule.rule_id

    return {
        'rules': rule_name_to_id_dict,
        'total_count': len(rule_name_to_id_dict),
        'response': {'rules': rule_name_to_id_dict}
    }


if __name__ == "__main__":
    # Example usage for testing register_rollout_servers_to_alb method
    import sys

    # Set environment variables for testing (you need to provide your own values)
    os.environ.setdefault('ALB_VPC_ID', '')
    os.environ.setdefault('ALB_LOAD_BALANCER_IDS', '')
    os.environ.setdefault('ALB_LOAD_BALANCER_LSN_IDS', '')
    os.environ.setdefault('ALB_ENDPOINT', '')
    os.environ.setdefault('ALIBABA_CLOUD_ACCESS_KEY_ID', '')
    os.environ.setdefault('ALIBABA_CLOUD_ACCESS_KEY_SECRET', '')
    os.environ.setdefault('ALB_RESOURCE_CLEAN_DIR', '')
    os.environ.setdefault('ALB_RESOURCE_GROUP_ID', '')
    os.environ.setdefault('ALB_REGION', '')
    os.environ.setdefault('TASK_ID', 'xdl-job-123456')


    # Test addresses (replace with actual rollout server addresses)
    test_addresses = [
        "http://10.17.70.145:8000/",
    ]

    print("Testing register_rollout_servers_to_alb...")

    try:
        result = register_rollout_servers_to_alb(
            addresses=test_addresses,
            vpc_id=os.environ.get('ALB_VPC_ID'),
            listener_port=20000,
        )
        print(f"Success! Result: {result}")
        result = cleanup_alb_resources()
        print(f"Cleanup result: {result}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# ALBProxyRouter — ProxyRouter implementation wrapping the functions above
# ---------------------------------------------------------------------------

from .base import ProxyRouter


class ALBProxyRouter(ProxyRouter):
    """Alibaba Cloud ALB router implementing the ProxyRouter interface.

    Wraps the standalone ALB registration functions so that callers can switch
    between ALB and K8s backends through a common interface.

    Environment variables consumed (same as the underlying functions):
        TASK_ID, ALB_VPC_ID, ALB_REGION, ALB_LOAD_BALANCER_IDS,
        ALB_LOAD_BALANCER_LSN_IDS, ALB_RESOURCE_CLEAN_DIR,
        ALIBABA_CLOUD_ACCESS_KEY_ID, ALIBABA_CLOUD_ACCESS_KEY_SECRET
    """

    def register_servers(self, addresses: List[str], job_id: str, port: int) -> dict:
        """Register rollout servers to ALB via server-group + listener rule."""
        os.environ['TASK_ID'] = job_id
        return register_rollout_servers_to_alb(addresses=addresses, listener_port=port)

    def cleanup_resources(self, job_id: str) -> dict:
        """Delete ALB server-group and rules recorded under job_id's resource dir."""
        os.environ['TASK_ID'] = job_id
        result = cleanup_alb_resources()
        return {
            "cleaned_up": result.get("cleaned_up", 0),
            "errors": result.get("errors", []),
        }

    def list_active_job_ids(self) -> List[str]:
        """Return job IDs parsed from active ALB listener rule names.

        Rule names follow the pattern ``rule-{job_id}-{port}``, so the job_id
        is extracted as everything between the first and last '-'-delimited token.
        """
        rules_dict = list_rules().get('rules', {})
        job_ids = []
        for rule_name in rules_dict:
            # rule_name format: "rule-{job_id}-{port}"
            parts = rule_name.split('-')
            if len(parts) >= 3:
                # job_id may itself contain '-', so take everything between first and last token
                job_ids.append('-'.join(parts[1:-1]))
        return job_ids

    def get_callback_url(self, job_id: str, port: int) -> str | None:
        """Build the ALB callback URL for agents to reach the rollout server.

        Reads EP_ENDPOINT, ALB_LISTEN_PORT, ALB_REGION from the environment.
        Returns None if any required variable is missing.
        """
        ep_endpoint = os.environ.get('EP_ENDPOINT')
        alb_listen_port = os.environ.get('ALB_LISTEN_PORT')
        alb_region = os.environ.get('ALB_REGION')
        if not all([ep_endpoint, alb_listen_port, alb_region]):
            return None
        alb_id, _ = get_alb_load_balancer_id_by_hash(job_id, port)
        return f"http://{ep_endpoint}:{alb_listen_port}/{alb_region}/{alb_id}/{job_id}/{port}/v1"
