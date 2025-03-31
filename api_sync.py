from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import logging

from auth import validate_device_token
from firestore_manager import FirestoreManager

router = APIRouter()
logger = logging.getLogger(__name__)
firestore = FirestoreManager()

class SyncRequest(BaseModel):
    device_id: str
    last_sync_time: str
    data: Dict[str, List[Dict[str, Any]]]

class SyncResponse(BaseModel):
    status: str
    sync_time: str
    data: Dict[str, List[Dict[str, Any]]]

@router.post("/sync", response_model=SyncResponse)
async def sync_device_data(
    request: SyncRequest,
    device_info: dict = Depends(validate_device_token)
):
    """
    Sync data between device and server:
    1. Receive changes from device
    2. Store those changes for other devices
    3. Send changes from other devices back to this device
    """
    try:
        # Verify device ownership
        if device_info["device_id"] != request.device_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to sync this device"
            )
        
        # Get user ID from the device token
        user_id = device_info.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Device not associated with user"
            )
        
        # Process data sent from device
        if request.data and any(len(records) > 0 for records in request.data.values()):
            try:
                for data_type, records in request.data.items():
                    for record in records:
                        await firestore.add_sync_record(
                            source_device_id=request.device_id,
                            user_id=user_id,
                            data_type=data_type,
                            data=record
                        )
                logger.info(f"Processed incoming sync data for device {request.device_id}, user {user_id}")
            except Exception as e:
                logger.error(f"Error processing incoming sync data: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                    detail=f"Failed to process incoming data: {str(e)}"
                )
        
        # Get data to send back to device
        try:
            data_for_device = await firestore.get_sync_records_for_device(
                request.device_id,
                user_id,
                request.last_sync_time
            )
            
            # Current time for the device to use as last_sync_time in next request
            current_time = datetime.utcnow().isoformat()
            
            return {
                "status": "success",
                "sync_time": current_time,
                "data": data_for_device
            }
        except Exception as e:
            logger.error(f"Error retrieving sync data: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"Failed to retrieve sync data: {str(e)}"
            )
    except HTTPException:
        # Re-raise HTTP exceptions to preserve status codes
        raise
    except Exception as e:
        logger.error(f"Sync operation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Sync failed due to server error: {str(e)}"
        )

@router.get("/pending")
async def get_pending_sync(
    device_id: str,
    device_info: dict = Depends(validate_device_token)
):
    """Get all pending sync records for a device"""
    try:
        # Verify device ownership
        if device_info["device_id"] != device_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this device's data"
            )
        
        user_id = device_info.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Device not associated with user"
            )
        
        # Fetch last sync time from device record
        device = await firestore.get_device(device_id)
        last_sync_time = device.get("last_sync_time", "1970-01-01T00:00:00")
        
        # Get pending records
        records = await firestore.get_sync_records_for_device(
            device_id,
            user_id,
            last_sync_time
        )
        
        return records
    except HTTPException:
        # Re-raise HTTP exceptions to preserve status codes
        raise
    except Exception as e:
        logger.error(f"Failed to get pending records: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve pending records: {str(e)}"
        )

@router.post("/mark-synced")
async def mark_synced(
    device_id: str,
    record_ids: List[str],
    device_info: dict = Depends(validate_device_token)
):
    """Mark records as synced after successful sync"""
    try:
        # Verify device ownership
        if device_info["device_id"] != device_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to modify this device's data"
            )
        
        user_id = device_info.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Device not associated with user"
            )
        
        # Update device's last sync time
        current_time = datetime.utcnow().isoformat()
        await firestore.update_device_sync_time(device_id, current_time)
        
        # Mark records as synced
        for record_id in record_ids:
            await firestore.mark_sync_record_as_synced(record_id, device_id)
        
        return {"status": "success", "message": f"Marked {len(record_ids)} records as synced"}
    except HTTPException:
        # Re-raise HTTP exceptions to preserve status codes
        raise
    except Exception as e:
        logger.error(f"Failed to mark records as synced: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to mark records as synced: {str(e)}"
        )