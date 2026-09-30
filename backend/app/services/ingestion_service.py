import logging
from datetime import datetime, timezone
import httpx
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.station import Station, StationStatus
from app.models.reading import Reading, ReadingSource
from app.models.anomaly import Anomaly, FaultType, AnomalyStatus
from app.models.correction import Correction
from app.models.sensor_health import SensorHealth
from app.models.fault_history import FaultHistory
import sys
import os
import math

logger = logging.getLogger(__name__)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def fetch_weather_api(lat: float, lon: float):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,surface_pressure&timezone=UTC"
    response = httpx.get(url, timeout=10.0)
    response.raise_for_status()
    data = response.json()
    current = data.get("current", {})
    return {
        "time": current.get("time"),
        "temperature": current.get("temperature_2m"),
        "humidity": current.get("relative_humidity_2m"),
        "pressure": current.get("surface_pressure"),
    }

import csv
CSV_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/College_station_Data.csv"))
STATE_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/csv_state.txt"))

def fetch_csv_row():
    idx = 0
    if os.path.exists(STATE_FILE_PATH):
        try:
            with open(STATE_FILE_PATH, "r") as f:
                idx = int(f.read().strip())
        except ValueError:
            pass
            
    if not os.path.exists(CSV_FILE_PATH):
        raise FileNotFoundError(f"CSV file not found at {CSV_FILE_PATH}")
        
    with open(CSV_FILE_PATH, "r") as f:
        reader = list(csv.DictReader(f))
        
    if not reader:
        raise ValueError("CSV is empty")
        
    if idx >= len(reader):
        logger.info(f"Reached end of CSV ({len(reader)} rows). Looping back to row 0.")
        print(f"Reached end of CSV ({len(reader)} rows). Looping back to row 0.")
        idx = 0
        
    row = reader[idx]
    
    with open(STATE_FILE_PATH, "w") as f:
        f.write(str(idx + 1))
        
    print(f"Consumed CSV Row {idx}: Temp={row['Temperature_C']} Hum={row['Humidity']} Press={row['Pressure_Hpa']}")
    logger.info(f"Consumed CSV Row {idx}: Temp={row['Temperature_C']} Hum={row['Humidity']} Press={row['Pressure_Hpa']}")
        
    return {
        "temperature": float(row["Temperature_C"]),
        "humidity": float(row["Humidity"]),
        "pressure": float(row["Pressure_Hpa"])
    }

def poll_stations():
    db: Session = SessionLocal()
    try:
        stations = db.query(Station).all()
        now = datetime.now(timezone.utc)
        
        # Load ML Pipeline
        try:
            import os
            from skyguard.main_pipeline import SkyGuardPipeline
            
            pipeline = SkyGuardPipeline()
            model_dir = os.path.join(os.path.dirname(__file__), "../../skyguard/models")
            fc_path = os.path.join(model_dir, "classifier.pkl")
            temp_path = os.path.join(model_dir, "temporal")
            temp_sklearn_path = os.path.join(model_dir, "temporal_sklearn.pkl")
            
            print(f"--- MODEL LOAD INFO ---")
            print(f"TORCH_AVAILABLE: {pipeline.temporal_ai.use_pytorch}")
            if os.path.exists(fc_path):
                print(f"classifier.pkl size: {os.path.getsize(fc_path)} bytes, mtime: {os.path.getmtime(fc_path)}")
            if os.path.exists(temp_sklearn_path):
                print(f"temporal_sklearn.pkl size: {os.path.getsize(temp_sklearn_path)} bytes, mtime: {os.path.getmtime(temp_sklearn_path)}")
            
            fc_loaded = pipeline.fault_classifier.load(fc_path)
            temp_loaded = pipeline.temporal_ai.load(temp_path)
            
            if not (fc_loaded and temp_loaded):
                logger.error("Failed to load models from disk!")
                return
                
        except ImportError as e:

            logger.error(f"Failed to import SkyGuardPipeline: {e}")
            return
            
        for station in stations:
            reading_data = None
            if station.is_primary:
                try:
                    real_data = fetch_weather_api(station.latitude, station.longitude)
                    data = fetch_csv_row()
                    reading_data = {
                        "temperature": data.get("temperature"),
                        "humidity": data.get("humidity"),
                        "pressure": data.get("pressure")
                    }
                except Exception as e:
                    logger.error(f"Error polling primary station {station.station_id}: {e}")
                    continue
            else:
                try:
                    # FETCH REAL METEO DATA (Ground Truth)
                    real_data = fetch_weather_api(station.latitude, station.longitude)
                    
                    # Simulated Sensor Data 
                    # (User provides primary sensor data, or we mock the other AWS stations with faults)
                    reading_data = {
                        "time": real_data.get("time"),
                        "temperature": real_data.get("temperature", 30.0),
                        "humidity": real_data.get("humidity", 50.0),
                        "pressure": real_data.get("pressure", 1000.0)
                    }
                    
                    import random
                    if station.station_id == "AWS-002":
                        # Drift Fault -> Faulty
                        reading_data["temperature"] += 4.5
                    elif station.station_id == "AWS-003":
                        # Genuine Event -> Warning (simulate sensor dropping)
                        reading_data["temperature"] -= 4.5
                    elif station.station_id == "AWS-004":
                        # Spike Fault -> Faulty
                        reading_data["temperature"] += 35.0
                            
                except Exception as e:
                    logger.error(f"Error fetching Open-Meteo for {station.station_id}: {e}")
                    continue
            
            try:
                if reading_data.get("temperature") is None:
                    raise ValueError("Sensor returned no temperature")
                    
                obs_time = now
                if not station.is_primary and reading_data.get("time"):
                    # Open-Meteo API returns ISO 8601 (e.g. "2023-01-01T12:00")
                    obs_time = datetime.fromisoformat(reading_data["time"])
                    if obs_time.tzinfo is not None:
                        obs_time = obs_time.replace(tzinfo=None)
                        
                    # Check for deduplication
                    existing = db.query(Reading).filter(
                        Reading.station_id == station.station_id,
                        Reading.timestamp == obs_time
                    ).first()
                    if existing:
                        print(f"[{station.station_id}] Skipped (Duplicate timestamp: {obs_time.isoformat()})")
                        continue
                        
                # Fetch actual history from DB BEFORE inserting the current reading
                past_readings = db.query(Reading).filter(
                    Reading.station_id == station.station_id,
                    Reading.timestamp < obs_time
                ).order_by(Reading.timestamp.desc()).limit(23).all()
                past_readings.reverse() # chronological
                
                reading = Reading(
                    station_id=station.station_id,
                    timestamp=obs_time,
                    temperature=reading_data["temperature"],
                    humidity=reading_data["humidity"],
                    pressure=reading_data["pressure"] or 1013.25,
                    source=ReadingSource.physical_sensor
                )
                db.add(reading)
                db.commit()
                db.refresh(reading)
                
                hist = []
                for r in past_readings:
                    hist.append({
                        "time": r.timestamp.isoformat(),
                        "temperature_2m": r.temperature,
                        "relative_humidity_2m": r.humidity,
                        "surface_pressure": r.pressure,
                        "pressure_msl": r.pressure + 20,
                        "station_id": station.station_id
                    })
                pipeline.recent_history = hist
                
                reading_dict = {
                    "time": reading.timestamp.isoformat(),
                    "temperature_2m": reading.temperature,
                    "relative_humidity_2m": reading.humidity,
                    "surface_pressure": reading.pressure,
                    "pressure_msl": reading.pressure + 20,
                    "station_id": station.station_id
                }
                
                # Compare Sensor Data against REAL Open-Meteo Data (as ground truth baseline)
                neighbors = {
                    "OpenMeteo-Truth": {
                        "temperature_2m": real_data.get("temperature", 30.0),
                        "relative_humidity_2m": real_data.get("humidity", 50.0),
                        "surface_pressure": real_data.get("pressure", 1000.0),
                        "pressure_msl": real_data.get("pressure", 1000.0) + 20
                    }
                }
                pipeline.neighbor_metadata = {
                    "OpenMeteo-Truth": {
                        "distance_km": 0.0,
                        "corr_temp": 1.0
                    }
                }
                diag = pipeline.process_reading(reading_dict, neighbors)
                
                diag_type = diag.get("diagnosis", {}).get("diagnosis_type", "NORMAL")
                out_fault = diag.get("anomaly_type")


                

                
                # We need the counterfactual vs classifier!
                # It's not exposed explicitly in the return dict unless we added it.
                # Actually, in main_pipeline.py: diag_res["classifier_verdict"] = fault_label
                # Let's extract it.
                cf_verdict = diag_type
                clf_verdict = diag.get("anomaly_type", "N/A")
                
                fault_type_enum = None
                if out_fault == "SPIKE": fault_type_enum = FaultType.Spike
                elif out_fault == "FROZEN": fault_type_enum = FaultType.Frozen
                elif out_fault == "DRIFT": fault_type_enum = FaultType.Drift
                elif out_fault == "COMM_FAILURE": fault_type_enum = FaultType.CommFailure

                if diag_type == "NORMAL":
                    status_enum = AnomalyStatus.resolved
                    station_status_enum = StationStatus.healthy
                elif diag_type == "GENUINE_EXTREME_EVENT":
                    status_enum = AnomalyStatus.warning
                    station_status_enum = StationStatus.warning
                else:
                    status_enum = AnomalyStatus.critical
                    station_status_enum = StationStatus.faulty
                    
                print(f"[{station.station_id}] T:{reading.temperature:.1f} P:{reading.pressure:.1f} H:{reading.humidity:.1f} | CF:{cf_verdict} | CLF:{clf_verdict} | Final:{station_status_enum.value}")
                
                station.status = station_status_enum
                db.commit()
                
                if diag_type != "NORMAL":
                    anomaly = Anomaly(
                        station_id=station.station_id,
                        reading_id=reading.id,
                        reading_timestamp=reading.timestamp,
                        detected_at=datetime.now(timezone.utc),
                        anomaly_score=diag.get('severity_score', 0.0),
                        fault_type=fault_type_enum,
                        status=status_enum,
                        temporal_evidence={"details": diag.get('evidence_metrics', {}).get("temporal_lstm_mse")},
                        physical_consistency={"details": diag.get('evidence_metrics', {}).get("physics_dew_point_c")},
                        spatial_evidence={"details": diag.get('evidence_metrics', {}).get("spatial_temp_z_score")}
                    )
                    db.add(anomaly)
                    db.commit()
                    db.refresh(anomaly)
                    
                    if status_enum == AnomalyStatus.critical:
                        ct = diag.get("corrected_telemetry", {})
                        if ct.get("needs_correction"):
                            correction = Correction(
                                anomaly_id=anomaly.id,
                                original_value=reading.temperature,
                                corrected_value=ct.get('temperature_2m'),
                                confidence=ct.get('reconstruction_confidence_pct', 90) / 100,
                                methodology={"method": ct.get('imputation_method')},
                                operator_decision="pending"
                            )
                            db.add(correction)
                            db.commit()
                            
                            fh = FaultHistory(
                                station_id=station.station_id,
                                component="Temperature Sensor",
                                action="imputed",
                                event_date=datetime.now(timezone.utc).date()
                            )
                            db.add(fh)
                            db.commit()
                        
                health = db.query(SensorHealth).filter(SensorHealth.station_id == station.station_id).first()
                if not health:
                    health = SensorHealth(station_id=station.station_id, fleet_health_score=100.0, mtbf_days=120)
                    db.add(health)
                
                sh = diag.get("sensor_health", {})
                health.fleet_health_score = sh.get("health_score_pct", 100.0)
                health.last_calculated = datetime.now(timezone.utc)
                db.commit()

            except Exception as e:
                logger.error(f"Error polling primary station {station.station_id}: {e}")
                db.rollback()
                continue
                
    except Exception as e:
        logger.error(f"Error in poll_stations job: {e}")
    finally:
        db.close()
