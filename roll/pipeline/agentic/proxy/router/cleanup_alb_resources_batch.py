#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch cleanup script for ALB resources.

This script lists all ALB rules, checks if the XDL job ID in each rule name
is in a cleanable state, and deletes the rule if it is.
"""

import os

from .alb_proxy_router import (
    ALBProxyRouter,
    create_alb_client,
    delete_resource_with_retry,
)
from .base import batch_cleanup


def batch_cleanup_alb_server_groups():
    """List all ALB server groups and delete them directly without validation."""
    print("Listing all ALB server groups...")

    try:
        resource_group_id = os.environ.get('ALB_RESOURCE_GROUP_ID')
        assert resource_group_id, "ALB_RESOURCE_GROUP_ID is not set"
        server_groups_response = list_server_groups(resource_group_id)
        server_groups_dict = server_groups_response.get('server_groups', {})

        if not server_groups_dict:
            print("No server groups found")
            return

        print(f"Found {len(server_groups_dict)} server groups")
        client = create_alb_client()
        deleted_count = 0

        for sg_name, sg_id in server_groups_dict.items():
            print(f"\nDeleting server group: {sg_name} ({sg_id})")
            try:
                success, error_msg = delete_resource_with_retry(client, 'server-group', sg_id, 5)
                if success:
                    print(f"Successfully deleted server group: {sg_name}")
                    deleted_count += 1
                else:
                    print(f"Failed to delete server group {sg_name}: {error_msg}")
            except Exception as e:
                print(f"Error deleting server group {sg_name}: {e}")
                continue

        print(f"\nFinished processing server groups. Deleted {deleted_count} server groups.")

    except Exception as e:
        print(f"Error listing or processing server groups: {e}")
        return


def list_server_groups(resource_group_id: str = None) -> dict:
    """
    List all Server Groups and return a mapping of server_group_name -> server_group_id.

    Only includes server groups not attached to any ALB.
    """
    from alibabacloud_alb20200616 import models as alb_20200616_models
    from alibabacloud_tea_util import models as util_models

    try:
        client = create_alb_client()
        max_results = 100
        all_server_groups = []
        next_token = None

        while True:
            request = alb_20200616_models.ListServerGroupsRequest(
                max_results=max_results,
                resource_group_id=resource_group_id
            )
            if next_token:
                request.next_token = next_token

            runtime = util_models.RuntimeOptions()
            resp = client.list_server_groups_with_options(request, runtime)

            if hasattr(resp.body, 'server_groups') and resp.body.server_groups:
                all_server_groups.extend(resp.body.server_groups)

            next_token = resp.body.next_token if hasattr(resp.body, 'next_token') else None
            if not next_token:
                break

        name_to_id = {}
        print(f"Found {len(all_server_groups)} server groups")
        for sg in all_server_groups:
            print(f"Server group: {sg.server_group_id} {sg.server_group_name} {sg.related_load_balancer_ids}")
            if sg.server_group_name and sg.server_group_id and sg.related_load_balancer_ids is None:
                name_to_id[sg.server_group_name] = sg.server_group_id

        print(f"Found {len(name_to_id)} server groups not related to any ALB")
        return {"server_groups": name_to_id}

    except Exception as error:
        print(f"Error listing server groups: {error}")
        if hasattr(error, 'message'):
            print(f"Error message: {error.message}")
        if hasattr(error, 'data') and error.data:
            print(f"Recommendation: {error.data.get('Recommend')}")
        return {"server_groups": {}}


def main():
    # Set environment variables for testing (you need to provide your own values)
    os.environ.setdefault('ALB_LOAD_BALANCER_LSN_IDS', '')
    os.environ.setdefault('ALB_ENDPOINT', '')
    os.environ.setdefault('ALIBABA_CLOUD_ACCESS_KEY_ID', '')
    os.environ.setdefault('ALIBABA_CLOUD_ACCESS_KEY_SECRET', '')
    os.environ.setdefault('ALB_RESOURCE_GROUP_ID', '')
    print("Starting batch ALB resource cleanup...")
    batch_cleanup(ALBProxyRouter())
    batch_cleanup_alb_server_groups()
    print("Batch cleanup completed")


if __name__ == "__main__":
    main()
