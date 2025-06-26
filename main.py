from datetime import datetime
import time
from fastapi import FastAPI, HTTPException
import pandas as pd

from journal_services import generate_journal_statistics_logic
from metrics_services import traiter, traiter_ewh

app = FastAPI()

def validate_dates(start_date: str, end_date: str):
    try:
        d1 = datetime.fromisoformat(start_date)
        d2 = datetime.fromisoformat(end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD or ISO format.")
    if d1 >= d2:
        raise HTTPException(status_code=400, detail="Start date must be strictly before end date.")
    
    return d1, d2

def clean_iso_date(raw_date: str) -> datetime:
    # نشيل أي كلمات قبل التاريخ
    raw_date = raw_date.strip()  # نشيل المسافات والأسطر الجديدة
    for word in raw_date.split():
        if 'T' in word and 'Z' in word:
            try:
                return datetime.fromisoformat(word.replace("Z", "+00:00"))
            except ValueError:
                continue
    raise ValueError(f"Invalid date format: {raw_date}")


@app.get("/journal")
def get_journal(cbox, start_date, end_date, threshold, is_detailed):
    threshold = int(threshold)
    is_detailed = bool(int(is_detailed))
    result = generate_journal_statistics_logic(cbox,start_date, end_date, threshold, is_detailed)
    return result


@app.get("/metrics")
def get_metrics(cbox, start_date, end_date, only_ewh = 0):
    # dates = validate_dates(start_date, end_date)
    # print("start date ", dates[0])
    # print("end date ", dates[1])
    only_ewh = bool(int(only_ewh))
    # start_date = datetime.fromisoformat(start_date.strip().replace("Z", "+00:00"))
    # end_date = datetime.fromisoformat(end_date.strip().replace("Z", "+00:00"))
    # start_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S.%fZ")
    # end_date = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S.%fZ")
    # print(start_date)
    # print(end_date)
    cbox = cbox.strip()
    start_date = start_date.strip()
    end_date = end_date.strip()

    if only_ewh:
        res =  traiter_ewh(cbox, start_date, end_date)
    else:
        res =  traiter(cbox, start_date, end_date)
    return res
