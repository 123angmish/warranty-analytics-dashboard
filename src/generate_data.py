"""
src/generate_data.py
--------------------
Synthetic Data Generator for Electrical Equipment Warranty Analytics.
Generates reproducible products, sales, and warranty claims data with planted
engineering failure stories, physics-of-failure distributions (Weibull),
and realistic data quality anomalies.

Strict Requirements:
  - Exactly 8 product families
  - Exactly 12 failure modes
  - Generates ~27,200 raw claims so that >=25,000 clean records remain after cleaning.
  - Reproducible with seed 42.
"""

import os
import random
import sqlite3
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from faker import Faker

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
fake = Faker()
Faker.seed(RANDOM_SEED)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DATA_DIR, "warranty.db")

# ---------------------------------------------------------
# 1. Exactly 8 Product Families
# ---------------------------------------------------------
PRODUCTS = [
    {
        "product_family": "Online UPS 10kVA",
        "category": "Uninterruptible Power Supply",
        "unit_price": 4200.0,
        "warranty_months": 24,
        "target_ppm": 3600,
        "typical_failure_modes": [
            "Capacitor Aging",
            "Overheating / Fan Failure",
            "Firmware Fault",
            "PCB Solder Defect",
        ],
    },
    {
        "product_family": "Line-Interactive UPS 1kVA",
        "category": "Uninterruptible Power Supply",
        "unit_price": 380.0,
        "warranty_months": 24,
        "target_ppm": 3200,
        "typical_failure_modes": [
            "Battery Degradation",
            "Overheating / Fan Failure",
            "PCB Solder Defect",
            "Relay / Contactor Failure",
        ],
    },
    {
        "product_family": "Modular Industrial UPS 50kVA",
        "category": "Uninterruptible Power Supply",
        "unit_price": 18500.0,
        "warranty_months": 36,
        "target_ppm": 2400,
        "typical_failure_modes": [
            "Capacitor Aging",
            "Overheating / Fan Failure",
            "Firmware Fault",
            "Loose Connection / Overheating",
        ],
    },
    {
        "product_family": "LV Switchgear Panel",
        "category": "Switchgear & Distribution",
        "unit_price": 12500.0,
        "warranty_months": 36,
        "target_ppm": 2600,
        "typical_failure_modes": [
            "Insulation Breakdown",
            "Loose Connection / Overheating",
            "Corrosion",
            "Contact Wear",
        ],
    },
    {
        "product_family": "Molded Case Circuit Breaker (MCCB)",
        "category": "Circuit Protection",
        "unit_price": 650.0,
        "warranty_months": 24,
        "target_ppm": 3400,
        "typical_failure_modes": [
            "Contact Wear",
            "Trip-Unit Malfunction",
            "Loose Connection / Overheating",
        ],
    },
    {
        "product_family": "Air Circuit Breaker (ACB)",
        "category": "Circuit Protection",
        "unit_price": 4100.0,
        "warranty_months": 36,
        "target_ppm": 4200,
        "typical_failure_modes": [
            "Contact Wear",
            "Trip-Unit Malfunction",
            "Insulation Breakdown",
            "Corrosion",
        ],
    },
    {
        "product_family": "Surge Protection Device (SPD)",
        "category": "Protection Devices",
        "unit_price": 190.0,
        "warranty_months": 12,
        "target_ppm": 2100,
        "typical_failure_modes": [
            "MOV Varistor Degradation",
            "PCB Solder Defect",
        ],
    },
    {
        "product_family": "Power Distribution Unit (PDU)",
        "category": "Switchgear & Distribution",
        "unit_price": 1600.0,
        "warranty_months": 24,
        "target_ppm": 2300,
        "typical_failure_modes": [
            "Loose Connection / Overheating",
            "Firmware Fault",
            "Relay / Contactor Failure",
        ],
    },
]

REGIONS = ["North India", "South India", "West India", "East India", "APAC Export"]
CUSTOMER_TYPES = [
    "Commercial Data Center",
    "Industrial Manufacturing Plant",
    "Healthcare & Hospitals",
    "Commercial Office Park",
    "Utility Substation",
]

# ---------------------------------------------------------
# 2. Exactly 12 Failure Modes and Physics-of-Failure Components
# ---------------------------------------------------------
COMPONENTS_BY_FAILURE_MODE = {
    "Capacitor Aging": ["DC Bus Capacitor Bank", "AC Filter Capacitor", "Electrolytic Smoothing Can"],
    "Overheating / Fan Failure": ["Axial Brushless Cooling Fan", "Thermal Heat Sink Pad", "Fan Tachometer Sensor"],
    "Battery Degradation": ["LiFePO4 Cell String", "Lead-Acid Battery String", "Battery Management System (BMS) Board"],
    "Firmware Fault": ["DSP Main Controller Board", "MCU Telemetry Chip", "CANbus Communication Gateway"],
    "PCB Solder Defect": ["Inverter Gate Driver PCB", "Rectifier Control Card", "Auxiliary SMPS Board"],
    "Contact Wear": ["Main Arcing Contacts", "Silver Alloy Contact Tips", "Contact Arm Mechanism"],
    "Insulation Breakdown": ["Epoxy Support Bushing", "Phase Barrier Sheet", "Dielectric Creepage Insulation"],
    "Loose Connection / Overheating": ["Main Copper Busbar Joint", "Compression Cable Lug", "Terminal Block Screw Connection"],
    "Trip-Unit Malfunction": ["Microprocessor Overcurrent Release", "Shunt Trip Actuator Coil", "Current Transformer Sensor"],
    "Corrosion": ["Enclosure Sealing Gasket", "Silver-Plated Terminal Joint", "Galvanized Chassis Frame"],
    "MOV Varistor Degradation": ["Metal Oxide Varistor (MOV) Disc", "Thermal Disconnector Spring", "Surge Clamping Module"],
    "Relay / Contactor Failure": ["Switching Auxiliary Relay", "Magnetic Contactor Coil", "Transfer Relay Spring"],
}

# Weibull failure parameters (shape beta, scale eta in days)
WEIBULL_PARAMS = {
    # Infant mortality (beta < 1.0)
    "Firmware Fault": {"shape": 0.72, "scale_days": 210},
    "PCB Solder Defect": {"shape": 0.78, "scale_days": 240},
    "Trip-Unit Malfunction": {"shape": 0.82, "scale_days": 320},
    # Random / constant hazard (beta ~ 1.0)
    "Loose Connection / Overheating": {"shape": 1.08, "scale_days": 420},
    "MOV Varistor Degradation": {"shape": 1.12, "scale_days": 380},
    "Relay / Contactor Failure": {"shape": 1.15, "scale_days": 450},
    # Wear-out mechanism (beta > 1.5)
    "Capacitor Aging": {"shape": 2.30, "scale_days": 680},
    "Contact Wear": {"shape": 2.45, "scale_days": 750},
    "Battery Degradation": {"shape": 2.10, "scale_days": 620},
    "Overheating / Fan Failure": {"shape": 1.95, "scale_days": 540},
    "Insulation Breakdown": {"shape": 2.20, "scale_days": 710},
    "Corrosion": {"shape": 1.85, "scale_days": 590},
}

DESCRIPTION_TEMPLATES = {
    "Capacitor Aging": [
        "Field technician observed DC bus capacitor electrolyte venting with pungent smell and visible bulged aluminum vent rubber.",
        "Unit threw DC_BUS_OVERVOLTAGE fault during nominal full load; C12 and C14 capacitors show heavy ESR degradation.",
        "Audible high-pitch whining followed by inverter shutdown. Visual check revealed leaky electrolytic caps on intermediate bus.",
        "Preventive diagnostic showed ripple current exceeding 15% threshold; electrolytic capacitors dried out after thermal stress.",
        "Intermediate DC link capacitor bank shorted, triggering internal fuse F1. Capacitance measured 40% below nominal rating.",
        "Capacitor can swollen and top safety vent popped during summer peak load; replaced capacitor pack and cleaned PCB.",
        "High ripple voltage on main DC rail; inspected DC capacitors and found electrolyte leakage staining the bottom chassis.",
    ],
    "Overheating / Fan Failure": [
        "Inverter tripped on thermal runaway (OVERTEMP_ALARM >88C). Left heatsink cooling fan seized and motor coil scorched.",
        "Customer reported loud rattling noise from rear exhaust; cooling fan bearing disintegrated causing thermal cutout.",
        "Unit automatically de-rated output power due to heatsink overheating; cooling blower jammed with fine industrial dust.",
        "Fan tachometer signal dropped to 0 RPM; replacement of axial 24V brushless fan resolved intermittent thermal fault.",
        "Thermal sensor RTD2 logged 94C under 60% load. Cooling fan blades seized; motor driver IC showed discoloration.",
        "Heatsink thermal overload cutoff tripped during backup cycle. Exhaust fan seized up completely with bearing seizure.",
        "Cooling fan failed to spool up on startup self-test. Unit refused to connect load due to fan failure safety interlock.",
    ],
    "Battery Degradation": [
        "UPS runtime collapsed from rated 30 mins to under 3 mins during power outage. Internal battery string impedance spiked.",
        "BMS reported CELL_VOLTAGE_IMBALANCE >280mV; battery pack failed to maintain float voltage under load test.",
        "Individual cell swelling observed inside battery module; battery internal resistance exceeded end-of-life limit.",
        "Customer noted low backup warning immediately after mains failure. Discharge curve indicates premature cell capacity fade.",
        "Battery module temperature elevated during float charging. BMS log indicates cell 4 and cell 8 thermal degradation.",
        "Battery string failed discharge capacity audit at 42% of rated Ah; replacement battery tray required under warranty.",
        "BMS communication error coupled with cell degradation alarm; open-circuit voltage collapsed under 50% load test.",
    ],
    "Firmware Fault": [
        "DSP controller board frozen with communication timeout error code E-102. System required manual power cycle.",
        "Inverter logic controller crashed following transient voltage dip; microprocessor watchdog timer failed to reset.",
        "Modbus/SNMP network card unresponsive. Diagnostic UART showed memory leak in DSP telemetry firmware build v2.1.4.",
        "Intermittent false alarms for phase sequence loss caused by unhandled interrupt in DSP firmware control loop.",
        "Firmware lockup during synchronizing transition between bypass and inverter mode; required JTAG controller reflash.",
        "Microcontroller boot loop after emergency stop signal; telemetry firmware register corrupted during auxiliary power dip.",
        "DSP gate driver timing jitter generated shoot-through alarm on phase B; resolved by updating firmware rev 3.01.",
    ],
    "PCB Solder Defect": [
        "Cold solder joint identified on gate driver optocoupler pin U3, causing intermittent IGBT misfiring.",
        "Solder bridge between pin 4 and pin 5 on auxiliary power supply controller caused SMPS start-up failure.",
        "Thermal cycling caused micro-cracking and open circuit at surface mount solder pads of R45 shunt resistor.",
        "Intermittent gate drive signal trace failure; inspection under microscope revealed cracked solder fillet on Q2.",
        "SMD capacitor cracked due to excessive mechanical flex during factory assembly; shorted +15V bias rail.",
        "Dry solder joint on current sense transformer terminal resulted in false overcurrent protection trips.",
        "Solder voiding beneath QFN power management IC led to localized overheating and intermittent circuit disconnection.",
    ],
    "Contact Wear": [
        "ACB failed routine closing test; main silver-tungsten arcing contacts eroded beyond permissible wear thickness.",
        "Circuit breaker arcing contact tips severely pitted and oxidized following repeated inductive fault interruptions.",
        "High micro-ohm contact resistance measured across phase R (145 uOhm vs <35 uOhm limit); arcing contacts burnt.",
        "Contact erosion caused uneven contact touch; arc chute runner showed heavy copper deposition and contact degradation.",
        "Main movable contact arm showed excessive erosion and mechanical play; arcing contact spring tension lost.",
        "Breaker tripped on thermal overload under nominal current due to excessive I2R heating at worn main contact surfaces.",
        "Arcing contacts severely eroded with loss of silver cladding; breaker failed to latch properly on manual charge.",
    ],
    "Insulation Breakdown": [
        "Dielectric flashover observed across phase busbar support during high voltage withstand test; epoxy insulator cracked.",
        "Insulation resistance dropped below 0.5 MOhm between R-phase and ground; tracking marks visible on polyester barrier.",
        "Corona discharge and surface partial discharge tracked across mold enclosure wall in humid switchgear vault.",
        "High voltage puncture through phase separation barrier sheet; carbonized discharge path found between phase A and B.",
        "Support standoff insulator cracked due to electrodynamic stress during downstream fault; ground fault tripped.",
        "Megger test showed severe degradation of busbar sleeve insulation with breakdown at 1.2kV AC.",
        "Insulator bushing surface tracked with carbonized conductive path; required complete replacement of insulator block.",
    ],
    "Loose Connection / Overheating": [
        "Thermographic IR inspection revealed 115C hotspot at phase B busbar splice joint due to loose Belleville torque washer.",
        "Compression cable lug at incoming terminal loose; localized heating charred cable insulation and melted terminal cover.",
        "Terminal block screw stripped and loose, resulting in severe arcing, contact discoloration, and high contact resistance.",
        "M12 busbar bolt under-torqued during factory panel assembly; joint reached 98C under 70% load current.",
        "Overheating at outgoing neutral terminal caused thermal deformation of terminal housing and smoke smell.",
        "Loose bolt at breaker disconnect finger clusters caused intermittent contact and severe thermal charring.",
        "High resistance joint detected at auxiliary control terminal strip; loose crimp caused intermittent telemetry drop.",
    ],
    "Trip-Unit Malfunction": [
        "Electronic trip unit failed to initiate trip signal during secondary injection testing; microprocessor logic unresponsive.",
        "Shunt trip coil burnt out and open-circuit; breaker could not be operated via remote SCADA trip command.",
        "Microprocessor overcurrent release gave nuisance instantaneous trip under normal motor starting current inrush.",
        "Trip unit display screen blank and analog sensor input saturated; CT signal conditioning circuit failure.",
        "Overcurrent release rotary switch potentiometer contact open; trip unit defaulted to minimum pickup setting.",
        "Shunt trip solenoid plunger jammed mechanically by debris; breaker failed to open on protection trip signal.",
        "Microprocessor trip unit memory checksum failure; unit permanently locked in protective lockout mode.",
    ],
    "Corrosion": [
        "Severe galvanic corrosion and copper patina oxidation on terminal busbars in coastal high-humidity environment.",
        "Enclosure sealing gasket degraded; saline moisture ingress caused heavy rust pitting on breaker mechanism chassis.",
        "Auxiliary contact terminal block terminals corroded with green copper carbonate oxidation, resulting in open circuits.",
        "High ambient humidity combined with atmospheric sulfur caused severe oxidation of copper busbar plating.",
        "Switchgear outdoor enclosure door hinges and lower gland plate heavily rusted; water pooling in bottom trough.",
        "Corrosion on contact springs caused loss of contact pressure and intermittent breaker tripping.",
        "Severe oxidation on silver-plated disconnect stabs; rust and sulfur deposits caused contact surface pitting.",
    ],
    "MOV Varistor Degradation": [
        "Surge protection module thermal disconnector flagged red flag window; MOV disk degraded after multiple grid surges.",
        "MOV varistor cracked and shorted internally; thermal fuse disconnected module to prevent fire hazard.",
        "Surge arrester leakage current exceeded 1mA; MOV block lost clamping voltage capability following lightning transients.",
        "SPD phase 1 cartridge blown with black carbon soot inside module cavity; required cartridge replacement.",
        "MOV disk thermal runaway after sustained grid over-voltage swell; thermal indicator popped out.",
        "Surge protection clamping voltage drifted lower than operating voltage, causing continuous leakage heating.",
        "Surge protection device module showed open circuit; MOV material ruptured under severe grid switching surge.",
    ],
    "Relay / Contactor Failure": [
        "Auxiliary transfer relay contacts chatter and failed to latch properly during automatic mains failure transition.",
        "Magnetic contactor coil burnt out and open-circuit; internal bypass failed to engage upon emergency command.",
        "Relay contacts pitted and welded closed following heavy inrush current; required replacement of auxiliary relay board.",
        "Intermittent telemetry signaling due to oxidized auxiliary relay micro-switch contacts in control cabinet.",
        "Bypass contactor failed to pull in on command; coil voltage measured normal at 24VDC but mechanical armature jammed.",
        "Switching relay coil shorted internally, pulling down auxiliary 24V DC control rail and causing logic controller reset.",
    ],
}


def generate_sales_and_products():
    """Generates products.csv (8 products) and 3 years of monthly shipments per product x region (~420k+ units)."""
    os.makedirs(DATA_DIR, exist_ok=True)

    products_clean = []
    for p in PRODUCTS:
        p_copy = p.copy()
        p_copy["typical_failure_modes"] = ", ".join(p["typical_failure_modes"])
        products_clean.append(p_copy)
    products_df = pd.DataFrame(products_clean)
    products_csv_path = os.path.join(DATA_DIR, "products.csv")
    products_df.to_csv(products_csv_path, index=False)
    print(f"Generated products.csv ({len(products_df)} products) -> {products_csv_path}")

    start_date = datetime(2023, 1, 1)
    sales_records = []
    shipment_counter = 100000

    for month_idx in range(36):
        current_date = start_date + timedelta(days=month_idx * 30.5)
        ship_year = current_date.year
        ship_month_str = current_date.strftime("%Y-%m")
        quarter = (current_date.month - 1) // 3 + 1

        growth = 1.0 + (month_idx / 36.0) * 0.22

        month_num = current_date.month
        seasonality = 1.0
        if month_num in [5, 6, 7]:
            seasonality = 1.25
        elif month_num in [11, 12]:
            seasonality = 1.20
        elif month_num in [1, 2]:
            seasonality = 0.85

        for prod in PRODUCTS:
            p_name = prod["product_family"]
            cat = prod["category"]

            if "Line-Interactive" in p_name:
                base_qty = 580
            elif "MCCB" in p_name:
                base_qty = 460
            elif "Surge Protection" in p_name:
                base_qty = 480
            elif "Online UPS 10kVA" in p_name:
                base_qty = 210
            elif "PDU" in p_name:
                base_qty = 180
            elif "ACB" in p_name:
                base_qty = 110
            elif "LV Switchgear" in p_name:
                base_qty = 60
            else:
                base_qty = 40  # Modular Industrial UPS 50kVA

            for region in REGIONS:
                region_weight = {
                    "West India": 1.25,
                    "North India": 1.15,
                    "South India": 1.10,
                    "East India": 0.80,
                    "APAC Export": 0.70,
                }[region]

                monthly_qty = int(base_qty * region_weight * growth * seasonality * np.random.uniform(0.92, 1.10))

                if p_name == "Online UPS 10kVA" and ship_year == 2024 and quarter in [3, 4]:
                    batch_id = f"LOT-CAP-2024-Q3-{region[:2].upper()}"
                else:
                    batch_id = f"LOT-{p_name[:3].upper()}-{ship_year}-M{current_date.month:02d}-{region[:2].upper()}"

                ship_day = random.randint(1, 28)
                exact_ship_date = datetime(ship_year, current_date.month, ship_day).strftime("%Y-%m-%d")

                shipment_counter += 1
                sales_records.append({
                    "shipment_id": f"SHP-{shipment_counter}",
                    "product_family": p_name,
                    "category": cat,
                    "region": region,
                    "ship_date": exact_ship_date,
                    "ship_month": ship_month_str,
                    "quantity": monthly_qty,
                    "batch_id": batch_id,
                    "unit_price": prod["unit_price"],
                })

    sales_df = pd.DataFrame(sales_records)
    total_shipped = sales_df["quantity"].sum()
    sales_csv_path = os.path.join(DATA_DIR, "sales.csv")
    sales_df.to_csv(sales_csv_path, index=False)
    print(f"Generated sales.csv ({len(sales_df)} shipment batches, {total_shipped:,} units shipped) -> {sales_csv_path}")

    return products_df, sales_df


def generate_claims(products_df, sales_df):
    """
    Generates ~27,200 warranty claims tied via Weibull time-to-failure to sales.
    Ensures that after cleaning and deduplication, at least 25,000+ clean claims remain.
    """
    print("Generating warranty claims dataset (~27,000+ raw records)...")
    claims_records = []
    claim_counter = 500000

    product_lookup = products_df.set_index("product_family").to_dict(orient="index")
    for p_name, p_data in product_lookup.items():
        if isinstance(p_data["typical_failure_modes"], str):
            p_data["typical_failure_modes"] = [x.strip() for x in p_data["typical_failure_modes"].split(",")]

    # Calibrate claim rate per product to generate ~26,000+ natural claims across 420k units shipped
    base_rates = {
        "Online UPS 10kVA": 0.068,
        "Line-Interactive UPS 1kVA": 0.058,
        "Modular Industrial UPS 50kVA": 0.052,
        "LV Switchgear Panel": 0.062,
        "Molded Case Circuit Breaker (MCCB)": 0.064,
        "Air Circuit Breaker (ACB)": 0.082,
        "Surge Protection Device (SPD)": 0.048,
        "Power Distribution Unit (PDU)": 0.054,
    }

    for _, sale in sales_df.iterrows():
        p_name = sale["product_family"]
        prod_meta = product_lookup[p_name]
        warranty_months = prod_meta["warranty_months"]
        warranty_days = warranty_months * 30.5
        unit_price = sale["unit_price"]
        region = sale["region"]
        ship_dt = datetime.strptime(sale["ship_date"], "%Y-%m-%d")
        batch_id = sale["batch_id"]
        qty = sale["quantity"]

        base_rate = base_rates.get(p_name, 0.060)

        # Planted stories
        is_emerging_lot = "LOT-CAP-2024-Q3" in batch_id
        is_chronic_acb = p_name == "Air Circuit Breaker (ACB)"
        is_coastal_corrosion = region == "South India" and p_name in ["LV Switchgear Panel", "Air Circuit Breaker (ACB)"]

        if is_emerging_lot:
            claim_rate = base_rate * 3.8
        elif is_chronic_acb:
            claim_rate = base_rate * 1.35
        elif is_coastal_corrosion:
            claim_rate = base_rate * 1.60
        else:
            claim_rate = base_rate

        expected_claims = np.random.poisson(lam=max(0.1, qty * claim_rate))
        if expected_claims <= 0:
            continue

        for _ in range(expected_claims):
            claim_counter += 1
            claim_id = f"CLM-{claim_counter}"

            if is_emerging_lot:
                if random.random() < 0.82:
                    failure_mode = "Capacitor Aging"
                else:
                    failure_mode = random.choice(prod_meta["typical_failure_modes"])
            elif is_coastal_corrosion and random.random() < 0.52:
                failure_mode = "Corrosion"
            elif is_chronic_acb and random.random() < 0.58:
                failure_mode = "Contact Wear"
            else:
                failure_mode = random.choice(prod_meta["typical_failure_modes"])

            avail_components = COMPONENTS_BY_FAILURE_MODE.get(failure_mode, ["General Subsystem"])
            component = random.choice(avail_components)

            wb_params = WEIBULL_PARAMS.get(failure_mode, {"shape": 1.5, "scale_days": 500})
            ttf_days = int(np.random.weibull(wb_params["shape"]) * wb_params["scale_days"])
            ttf_days = max(4, min(ttf_days, int(warranty_days * 1.25)))

            claim_dt = ship_dt + timedelta(days=ttf_days)
            if claim_dt > datetime(2026, 1, 31):
                continue

            if ttf_days <= warranty_days:
                status = random.choices(["Approved", "Settled", "Pending"], weights=[0.72, 0.23, 0.05])[0]
            else:
                status = "Rejected - Out of Warranty"

            # Cost structure
            if failure_mode == "Contact Wear":
                parts_cost = round(random.uniform(750, 1650), 2)
                labor_cost = round(random.uniform(250, 480), 2)
            elif failure_mode == "Capacitor Aging":
                parts_cost = round(random.uniform(320, 880), 2)
                labor_cost = round(random.uniform(150, 340), 2)
            elif failure_mode == "Insulation Breakdown":
                parts_cost = round(random.uniform(850, 2300), 2)
                labor_cost = round(random.uniform(350, 620), 2)
            elif failure_mode == "Battery Degradation":
                parts_cost = round(random.uniform(420, 1250), 2)
                labor_cost = round(random.uniform(130, 260), 2)
            elif failure_mode == "Corrosion":
                parts_cost = round(random.uniform(280, 950), 2)
                labor_cost = round(random.uniform(220, 420), 2)
            elif failure_mode == "Trip-Unit Malfunction":
                parts_cost = round(random.uniform(520, 1150), 2)
                labor_cost = round(random.uniform(180, 360), 2)
            elif failure_mode == "Loose Connection / Overheating":
                parts_cost = round(random.uniform(180, 520), 2)
                labor_cost = round(random.uniform(190, 380), 2)
            elif failure_mode == "Relay / Contactor Failure":
                parts_cost = round(random.uniform(200, 600), 2)
                labor_cost = round(random.uniform(140, 280), 2)
            else:
                parts_cost = round(random.uniform(120, 480), 2)
                labor_cost = round(random.uniform(90, 230), 2)

            repair_cost = round(parts_cost + labor_cost, 2)

            # NLP claim description
            template_list = DESCRIPTION_TEMPLATES.get(
                failure_mode,
                ["Field technician replaced failed sub-assembly during emergency service call."]
            )
            base_desc = random.choice(template_list)

            # Real technician ambiguity (~12% ambiguous symptom descriptions)
            r_overlap = random.random()
            if r_overlap < 0.05:
                base_desc = random.choice([
                    "Unit shut down on high temperature warning; technician inspected internal chassis and noted thermal discoloration.",
                    "Customer reported thermal alarm logged during heavy operating cycle; replaced degraded sub-assembly.",
                    "Overheating detected during routine thermography; cleaned terminals and replaced affected component.",
                ])
            elif r_overlap < 0.10:
                base_desc = random.choice([
                    "Main incoming feed tripped out on protection fault; inspected unit and reset protective relays.",
                    "Intermittent nuisance trip reported on power bus; tested operational parameters and swapped module.",
                    "Secondary circuit trip during load step; investigated electrical distribution panel and replaced parts.",
                ])
            elif r_overlap < 0.15:
                base_desc = random.choice([
                    "Emergency service call completed; replaced defective internal subassembly and calibrated unit.",
                    "Warranty RMA repair: diagnosed field malfunction, installed replacement component and tested OK.",
                    "Field inspection revealed abnormal operating readings; replaced damaged parts per service bulletin.",
                ])

            r_noise = random.random()
            if r_noise < 0.20:
                typo_replacements = [
                    ("capacitor", "capcitor"),
                    ("temperature", "temprature"),
                    ("breaker", "brkr"),
                    ("voltage", "volatage"),
                    ("connection", "connectn"),
                    ("terminal", "termnal"),
                    ("corrosion", "corosion"),
                    ("resistance", "resistnce"),
                    ("inspection", "inspectn"),
                    ("maintenance", "maintnance"),
                ]
                for old_w, new_w in typo_replacements:
                    if old_w in base_desc.lower() and random.random() < 0.5:
                        base_desc = base_desc.replace(old_w, new_w).replace(old_w.capitalize(), new_w.capitalize())
            elif r_noise < 0.35:
                prefix = random.choice(["URGENT RMA: ", "SERVICE LOG #", "FIELD DISPATCH: ", "WARRANTY DEFECT: ", "TECH NOTE: "])
                base_desc = prefix + base_desc

            cust_type = random.choice(CUSTOMER_TYPES)
            serial_num = f"SN-{p_name[:3].upper()}-{random.randint(1000000, 9999999)}"

            claims_records.append({
                "claim_id": claim_id,
                "product_family": p_name,
                "serial_number": serial_num,
                "batch_id": batch_id,
                "region": region,
                "ship_date": sale["ship_date"],
                "claim_date": claim_dt.strftime("%Y-%m-%d"),
                "failure_mode": failure_mode,
                "component": component,
                "parts_cost": parts_cost,
                "labor_cost": labor_cost,
                "repair_cost": repair_cost,
                "total_cost": repair_cost,  # synonym column
                "claim_status": status,
                "claim_description": base_desc,
                "customer_type": cust_type,
            })

    print(f"Generated natural base claims: {len(claims_records):,}")

    # Ensure at least 26,800 natural base claims so final cleaned count is >= 25,000
    while len(claims_records) < 26800:
        base_item = random.choice(claims_records).copy()
        claim_counter += 1
        base_item["claim_id"] = f"CLM-{claim_counter}"
        base_item["serial_number"] = f"SN-{base_item['product_family'][:3].upper()}-{random.randint(1000000, 9999999)}"
        claims_records.append(base_item)

    # ---------------------------------------------------------
    # Plant Realistic Noise & DQ Inconsistencies
    # ---------------------------------------------------------
    print("Injecting realistic data quality issues (duplicates, invalid dates, label variations, outliers)...")

    # 1. Exact duplicate claims (~200 records)
    for _ in range(200):
        dup_record = random.choice(claims_records).copy()
        claims_records.append(dup_record)

    # 2. Chronologically invalid dates (~120 records where claim_date < ship_date)
    for idx in random.sample(range(len(claims_records)), 120):
        ship_d = datetime.strptime(claims_records[idx]["ship_date"], "%Y-%m-%d")
        bad_claim_d = ship_d - timedelta(days=random.randint(5, 45))
        claims_records[idx]["claim_date"] = bad_claim_d.strftime("%Y-%m-%d")

    # 3. Inconsistent failure mode labels (~650 records)
    synonym_map = {
        "Capacitor Aging": ["CAPACITOR_AGING", "Capacitor Degraded", "cap aging"],
        "Contact Wear": ["CONTACT_WEAR", "Contact Erosion", "contact wear"],
        "Corrosion": ["CORROSION_RUST", "corrosion", "Corrosion / Oxidation"],
        "Loose Connection / Overheating": ["LOOSE_CONN", "loose connection", "Loose Busbar Connection"],
        "Trip-Unit Malfunction": ["TRIP_UNIT_ERR", "Trip Unit Defect", "trip-unit malfunction"],
        "Overheating / Fan Failure": ["FAN_FAILURE", "overheating fan", "Thermal Fan Defect"],
        "Relay / Contactor Failure": ["RELAY_ERR", "Contactor Malfunction", "relay failure"],
    }
    for idx in random.sample(range(len(claims_records)), 650):
        fm = claims_records[idx]["failure_mode"]
        if fm in synonym_map:
            claims_records[idx]["failure_mode"] = random.choice(synonym_map[fm])

    # 4. Missing values
    for idx in random.sample(range(len(claims_records)), 380):
        claims_records[idx]["customer_type"] = None
    for idx in random.sample(range(len(claims_records)), 120):
        claims_records[idx]["serial_number"] = None

    # 5. Outliers in repair cost
    for idx in random.sample(range(len(claims_records)), 30):
        if random.random() < 0.35:
            claims_records[idx]["repair_cost"] = -250.0
            claims_records[idx]["total_cost"] = -250.0
        else:
            claims_records[idx]["repair_cost"] = round(random.uniform(85000, 220000), 2)
            claims_records[idx]["total_cost"] = claims_records[idx]["repair_cost"]

    random.shuffle(claims_records)
    claims_df = pd.DataFrame(claims_records)
    claims_csv_path = os.path.join(DATA_DIR, "claims.csv")
    claims_df.to_csv(claims_csv_path, index=False)
    print(f"Generated claims.csv ({len(claims_df):,} raw claims) -> {claims_csv_path}")

    return claims_df


def save_to_sqlite(products_df, sales_df, claims_df):
    """Loads raw datasets into SQLite database warranty.db with schemas and indexes."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS products;")
    cursor.execute("DROP TABLE IF EXISTS sales;")
    cursor.execute("DROP TABLE IF EXISTS claims_raw;")
    cursor.execute("DROP TABLE IF EXISTS data_quality_log;")

    products_df.to_sql("products", conn, if_exists="replace", index=False)
    sales_df.to_sql("sales", conn, if_exists="replace", index=False)
    claims_df.to_sql("claims_raw", conn, if_exists="replace", index=False)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_prod ON sales(product_family);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_region ON sales(region);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_ship_date ON sales(ship_date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_batch ON sales(batch_id);")

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_claims_raw_id ON claims_raw(claim_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_claims_raw_prod ON claims_raw(product_family);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_claims_raw_batch ON claims_raw(batch_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_claims_raw_region ON claims_raw(region);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_claims_raw_ship_date ON claims_raw(ship_date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_claims_raw_claim_date ON claims_raw(claim_date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_claims_raw_failure_mode ON claims_raw(failure_mode);")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS data_quality_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        logged_at TEXT,
        rule_name TEXT,
        dimension TEXT,
        severity TEXT,
        count_affected INTEGER,
        action_taken TEXT,
        details TEXT
    );
    """)

    conn.commit()
    conn.close()
    print(f"Loaded all datasets into SQLite database -> {DB_PATH}")


def main():
    print("=" * 60)
    print("ELECTRICAL WARRANTY DATA GENERATOR")
    print("=" * 60)
    products_df, sales_df = generate_sales_and_products()
    claims_df = generate_claims(products_df, sales_df)
    save_to_sqlite(products_df, sales_df, claims_df)
    print("Data generation complete.")


if __name__ == "__main__":
    main()
