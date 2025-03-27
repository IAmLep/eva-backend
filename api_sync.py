from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel

from database import (
    get_db, 
    get_pending_sync_records, 
    mark_records_as_synced,
    process_incoming_sync_data, 
    get_sync_data_for_device
)

router = APIRouter()

class SyncRequest(BaseModel):
    device_id: str
    last_sync_time: str
    data: Dict[str, List[Dict[str, Any]]]

class SyncResponse(BaseModel):
    status: str
    sync_time: str
    data: Dict[str, List[Dict[str, Any]]]

@router.post("/sync", response_model=SyncResponse)
def sync_device_data(request: SyncRequest, db: Session = Depends(get_db)):
    """
    Sync data between device and server:
    1. Receive changes from device
    2. Store those changes for other devices
    3. Send changes from other devices back to this device
    """
    try:
        # Process data sent from device
        if request.data and any(len(records) > 0 for records in request.data.values()):
            process_incoming_sync_data(db, request.data, request.device_id)
        
        # Get data to send back to device
        data_for_device = get_sync_data_for_device(
            db, 
            request.device_id, 
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
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

@router.get("/pending", response_model=Dict[str, List[Dict[str, Any]]])
def get_pending_sync(device_id: str, db: Session = Depends(get_db)):
    """Get all pending sync records for a device"""
    try:
        return get_pending_sync_records(db, device_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pending records: {str(e)}")

@router.post("/mark-synced")
def mark_synced(
    device_id: str, 
    records: Dict[str, List[Dict[str, Any]]], 
    db: Session = Depends(get_db)
):
    """Mark records as synced after successful sync"""
    try:
        mark_records_as_synced(db, records)
        return {"status": "success", "message": "Records marked as synced"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to mark records: {str(e)}")