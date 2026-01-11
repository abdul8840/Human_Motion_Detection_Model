# picam_truth_intelligence.py
"""
PICAM — BEHAVIORAL TRUTH INTELLIGENCE SYSTEM v2.0
Not a CCTV system. Not a bottleneck detector. Not a dashboard.
Exists to expose hidden operational truth using human behavior, time, motion, and absence.
If something looks normal but leaks money, trust, or opportunity — we surface it.
"""

import csv
import os
import numpy as np
from datetime import datetime, date, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import json

class BehavioralTruthIntelligence:
    """
    PICAM Behavioral Truth Intelligence System v2.0
    
    CORE MISSION: Answer ONE question every day:
    "Where did we lose value today — and why?"
    
    Value may be: Revenue, Attention, Trust, Control, Discipline, Opportunity
    """
    
    def __init__(self, reports_dir: str = "picam_reports"):
        self.reports_dir = reports_dir
        self.today = date.today()
        self.today_str = self.today.strftime("%Y%m%d")
        
        # Fundamental Beliefs (Hard Rules)
        self.beliefs = {
            1: "Absence is more dangerous than presence",
            2: "Speed ≠ Efficiency",
            3: "Humans lie, behavior doesn't",
            4: "Owners don't need data — they need truth"
        }
        
        # Advanced Behavioral Patterns to Detect
        self.patterns = {
            'opportunity_loss': {
                'name': 'OPPORTUNITY LOSS',
                'pattern': 'High entrance activity + Low dwell at counter/table + Short interactions',
                'truth': 'People came. Nobody caught them.',
                'value_loss': ['Revenue', 'Attention', 'Trust']
            },
            'responsibility_failure': {
                'name': 'RESPONSIBILITY FAILURE',
                'pattern': 'Unattended alerts + Repeated across time + Activity elsewhere continues',
                'truth': 'Duty exists on paper, not in reality.',
                'value_loss': ['Control', 'Discipline', 'Trust']
            },
            'chaos_windows': {
                'name': 'CHAOS WINDOWS',
                'pattern': 'Bursts of rapid motion + Multiple alerts close in time + Short stays, frequent exits',
                'truth': 'System lost control temporarily.',
                'value_loss': ['Control', 'Attention', 'Opportunity']
            },
            'false_calm': {
                'name': 'FALSE CALM (MOST DANGEROUS)',
                'pattern': 'No bottlenecks + No long waits + Low dwell + exits',
                'truth': 'Silent loss. Owner feels safe. Money leaks quietly.',
                'value_loss': ['Revenue', 'Trust', 'Opportunity']
            }
        }
        
        self.data = {}
        self.insights = []
        self.value_loss = defaultdict(int)
        
        print(f"\n{'='*80}")
        print("PICAM — BEHAVIORAL TRUTH INTELLIGENCE SYSTEM v2.0")
        print("="*80)
        print("CORE MISSION: 'Where did we lose value today — and why?'")
        print("\nFUNDAMENTAL BELIEFS:")
        for key, belief in self.beliefs.items():
            print(f"  {key}. {belief}")
        print("="*80)
    
    def load_and_analyze(self) -> bool:
        """Load all data and perform behavioral analysis."""
        print(f"\n📊 LOADING TRUTH DATA")
        print(f"Date: {self.today.strftime('%Y-%m-%d')}")
        print("-"*40)
        
        # Load all CSV files
        self.data = self._load_all_data()
        
        if not self.data['detections']:
            print("❌ No detection data found.")
            print("   Run 'python picam_with_csv.py' first to collect behavioral data.")
            return False
        
        print(f"✓ Detections: {len(self.data['detections'])} behavioral markers")
        print(f"✓ Motion Events: {len(self.data['motion_events'])} activity samples")
        print(f"✓ Alerts: {len(self.data['alerts'])} system warnings")
        print(f"✓ Zones: {len(self.data['zone_stats'])} monitored areas")
        
        # Perform behavioral analysis
        print(f"\n🧠 ANALYZING BEHAVIORAL PATTERNS")
        print("-"*40)
        
        self._analyze_behavioral_patterns()
        
        return True
    
    def _load_all_data(self) -> Dict[str, List]:
        """Load all behavioral data from CSV files."""
        data = {
            'detections': [],
            'motion_events': [],
            'alerts': [],
            'zone_stats': [],
            'daily_stats': {}
        }
        
        # File mappings
        file_mappings = {
            'detections': f"detections_{self.today_str}.csv",
            'motion_events': f"motion_events_{self.today_str}.csv",
            'alerts': f"alerts_{self.today_str}.csv",
            'zone_stats': f"zone_stats_{self.today_str}.csv",
            'daily_stats': f"daily_stats_{self.today_str}.csv"
        }
        
        for data_type, filename in file_mappings.items():
            filepath = os.path.join(self.reports_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        if data_type == 'daily_stats':
                            reader = csv.DictReader(f)
                            for row in reader:
                                if row['date'] == self.today.strftime("%Y-%m-%d"):
                                    data['daily_stats'] = row
                                    break
                        else:
                            reader = csv.DictReader(f)
                            data[data_type] = list(reader)
                    
                    print(f"   ✓ {filename}: {len(data[data_type]) if isinstance(data[data_type], list) else 1} records")
                except Exception as e:
                    print(f"   ⚠ {filename}: Error - {e}")
            else:
                print(f"   ⚠ {filename}: Not found")
        
        return data
    
    def _analyze_behavioral_patterns(self):
        """Analyze data for behavioral patterns using CAUSE → EFFECT → LOSS framework."""
        
        # Create hourly behavioral analysis
        hourly_behavior = self._create_hourly_behavioral_analysis()
        
        print("   Asking behavioral truth questions...")
        
        # 1. OPPORTUNITY LOSS Analysis
        self._detect_opportunity_loss(hourly_behavior)
        
        # 2. RESPONSIBILITY FAILURE Analysis
        self._detect_responsibility_failure()
        
        # 3. CHAOS WINDOWS Analysis
        self._detect_chaos_windows()
        
        # 4. FALSE CALM Analysis
        self._detect_false_calm(hourly_behavior)
        
        # 5. Additional behavioral questions
        self._ask_behavioral_questions(hourly_behavior)
    
    def _create_hourly_behavioral_analysis(self) -> Dict:
        """Create hourly analysis focusing on behavioral patterns."""
        hourly_data = defaultdict(lambda: {
            'entrances': 0,
            'counter_dwell': [],  # Times spent at counter
            'table_dwell': [],    # Times spent at table
            'motion_intensity': [],
            'interaction_count': 0,
            'exit_count': 0,
            'alerts': [],
            'zones_active': set(),
            'faces_seen': 0,
            'rapid_motion_count': 0
        })
        
        # Process detections for behavioral markers
        last_seen = {}  # Track last seen time per object for dwell calculation
        
        for detection in self.data['detections']:
            try:
                timestamp = datetime.strptime(
                    detection['timestamp'].split('.')[0],
                    "%Y-%m-%d %H:%M:%S"
                )
                hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
                
                obj_id = detection['object_id']
                obj_type = detection['object_type']
                zone = detection.get('zone_name', '')
                
                # Count faces (human attention markers)
                if obj_type == 'face':
                    hourly_data[hour_key]['faces_seen'] += 1
                
                # Track entrances
                if zone == 'Entrance':
                    hourly_data[hour_key]['entrances'] += 1
                
                # Track dwell time (simplified)
                if zone in ['Counter', 'Table']:
                    dwell_key = f"{obj_id}_{zone}"
                    if dwell_key in last_seen:
                        dwell_seconds = (timestamp - last_seen[dwell_key]).total_seconds()
                        if zone == 'Counter':
                            hourly_data[hour_key]['counter_dwell'].append(dwell_seconds)
                        else:
                            hourly_data[hour_key]['table_dwell'].append(dwell_seconds)
                    last_seen[dwell_key] = timestamp
                
                # Track active zones
                if zone:
                    hourly_data[hour_key]['zones_active'].add(zone)
                    
            except Exception:
                continue
        
        # Process motion events for activity intensity
        for motion_event in self.data['motion_events']:
            try:
                timestamp = datetime.strptime(
                    motion_event['timestamp'].split('.')[0],
                    "%Y-%m-%d %H:%M:%S"
                )
                hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
                
                motion_score = float(motion_event.get('motion_score', 0))
                hourly_data[hour_key]['motion_intensity'].append(motion_score)
                
                if motion_event.get('rapid_motion') == 'True':
                    hourly_data[hour_key]['rapid_motion_count'] += 1
                    
            except Exception:
                continue
        
        # Process alerts for system warnings
        for alert in self.data['alerts']:
            try:
                timestamp = datetime.strptime(
                    alert['timestamp'].split('.')[0],
                    "%Y-%m-%d %H:%M:%S"
                )
                hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
                hourly_data[hour_key]['alerts'].append(alert['alert_type'])
            except Exception:
                continue
        
        # Calculate behavioral metrics
        for hour_key, data in hourly_data.items():
            # Average counter dwell time
            if data['counter_dwell']:
                data['avg_counter_dwell'] = np.mean(data['counter_dwell'])
                data['counter_interaction_rate'] = len([d for d in data['counter_dwell'] if d > 60]) / max(1, data['faces_seen'])
            else:
                data['avg_counter_dwell'] = 0
                data['counter_interaction_rate'] = 0
            
            # Motion intensity
            if data['motion_intensity']:
                data['avg_motion'] = np.mean(data['motion_intensity'])
                data['motion_variance'] = np.var(data['motion_intensity'])
            else:
                data['avg_motion'] = 0
                data['motion_variance'] = 0
            
            # Alert density
            data['alert_density'] = len(data['alerts']) / max(1, data['entrances'])
            
            # Zone utilization
            data['zones_used'] = len(data['zones_active'])
        
        return dict(hourly_data)
    
    def _detect_opportunity_loss(self, hourly_data: Dict):
        """Detect Pattern 1: OPPORTUNITY LOSS"""
        print("   • Checking for missed engagements...")
        
        opportunity_hours = []
        
        for hour, data in hourly_data.items():
            # Pattern: High entrance but low meaningful interaction
            if (data['entrances'] >= 5 and  # Reasonable footfall
                data['avg_counter_dwell'] < 45 and  # Less than 45 seconds average
                data['counter_interaction_rate'] < 0.3):  # Less than 30% meaningful interactions
                
                hour_str = hour.strftime('%H:%M')
                opportunity_hours.append({
                    'hour': hour_str,
                    'entrances': data['entrances'],
                    'avg_dwell': data['avg_counter_dwell'],
                    'interaction_rate': data['counter_interaction_rate'],
                    'severity': 'HIGH' if data['entrances'] > 10 else 'MEDIUM'
                })
        
        if opportunity_hours:
            insight = {
                'pattern': 'opportunity_loss',
                'title': 'PEOPLE CAME, NOBODY CAUGHT THEM',
                'evidence': f"{len(opportunity_hours)} hours showed high entrance activity but low engagement",
                'value_loss': ['Revenue', 'Attention', 'Trust'],
                'examples': opportunity_hours[:3],  # Top 3 examples
                'confidence': 'HIGH' if len(opportunity_hours) >= 3 else 'MEDIUM',
                'truth_statement': self.patterns['opportunity_loss']['truth']
            }
            self.insights.append(insight)
            self.value_loss['Revenue'] += len(opportunity_hours)
            print(f"     ⚠ Detected in {len(opportunity_hours)} hours")
    
    def _detect_responsibility_failure(self):
        """Detect Pattern 2: RESPONSIBILITY FAILURE"""
        print("   • Checking for unclaimed duties...")
        
        unattended_alerts = [a for a in self.data['alerts'] 
                           if a.get('alert_type') == 'unattended_station']
        idle_alerts = [a for a in self.data['alerts'] 
                      if a.get('alert_type') == 'idle_staff']
        
        # Check for repeated alerts without correction
        alert_times = []
        for alert in unattended_alerts + idle_alerts:
            try:
                timestamp = datetime.strptime(
                    alert['timestamp'].split('.')[0],
                    "%Y-%m-%d %H:%M:%S"
                )
                alert_times.append(timestamp)
            except:
                continue
        
        alert_times.sort()
        
        # Find clusters of alerts (responsibility gaps)
        clusters = []
        if alert_times:
            current_cluster = [alert_times[0]]
            for i in range(1, len(alert_times)):
                time_diff = (alert_times[i] - alert_times[i-1]).total_seconds() / 60
                if time_diff <= 30:  # Within 30 minutes
                    current_cluster.append(alert_times[i])
                else:
                    if len(current_cluster) >= 3:
                        clusters.append(current_cluster)
                    current_cluster = [alert_times[i]]
            
            if len(current_cluster) >= 3:
                clusters.append(current_cluster)
        
        if clusters:
            insight = {
                'pattern': 'responsibility_failure',
                'title': 'DUTY ON PAPER, NOT IN REALITY',
                'evidence': f"{len(clusters)} responsibility gaps detected",
                'value_loss': ['Control', 'Discipline', 'Trust'],
                'examples': [f"Cluster of {len(c)} alerts in {(c[-1] - c[0]).total_seconds()/60:.0f}min" 
                           for c in clusters[:2]],
                'confidence': 'HIGH' if len(clusters) >= 2 else 'MEDIUM',
                'truth_statement': self.patterns['responsibility_failure']['truth']
            }
            self.insights.append(insight)
            self.value_loss['Control'] += len(clusters)
            print(f"     ⚠ {len(clusters)} responsibility gaps")
    
    def _detect_chaos_windows(self):
        """Detect Pattern 3: CHAOS WINDOWS"""
        print("   • Checking for control loss moments...")
        
        # Analyze motion events for chaos patterns
        rapid_motion_hours = defaultdict(int)
        
        for motion_event in self.data['motion_events']:
            try:
                timestamp = datetime.strptime(
                    motion_event['timestamp'].split('.')[0],
                    "%Y-%m-%d %H:%M:%S"
                )
                hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
                
                if motion_event.get('rapid_motion') == 'True':
                    rapid_motion_hours[hour_key] += 1
            except:
                continue
        
        # Find hours with excessive rapid motion (chaos indicators)
        chaos_hours = []
        for hour, count in rapid_motion_hours.items():
            if count >= 5:  # 5+ rapid motion events in an hour
                hour_str = hour.strftime('%H:%M')
                chaos_hours.append({
                    'hour': hour_str,
                    'rapid_motions': count,
                    'severity': 'HIGH' if count >= 10 else 'MEDIUM'
                })
        
        if chaos_hours:
            insight = {
                'pattern': 'chaos_windows',
                'title': 'SYSTEM LOST CONTROL TEMPORARILY',
                'evidence': f"{len(chaos_hours)} hours showed loss of control patterns",
                'value_loss': ['Control', 'Attention', 'Opportunity'],
                'examples': chaos_hours[:2],
                'confidence': 'HIGH' if len(chaos_hours) >= 2 else 'MEDIUM',
                'truth_statement': self.patterns['chaos_windows']['truth']
            }
            self.insights.append(insight)
            self.value_loss['Control'] += len(chaos_hours)
            print(f"     ⚠ {len(chaos_hours)} chaos windows")
    
    def _detect_false_calm(self, hourly_data: Dict):
        """Detect Pattern 4: FALSE CALM (Most Dangerous)"""
        print("   • Checking for silent value leaks...")
        
        false_calm_hours = []
        
        for hour, data in hourly_data.items():
            # Pattern: Activity but no engagement, no alerts, low dwell
            if (data['entrances'] > 0 and  # Some activity
                data['avg_counter_dwell'] < 30 and  # Very brief interactions
                len(data['alerts']) == 0 and  # No system warnings
                data['counter_interaction_rate'] < 0.2 and  # Minimal meaningful interaction
                data['motion_variance'] < 0.01):  # Consistent low-level activity
                
                hour_str = hour.strftime('%H:%M')
                false_calm_hours.append({
                    'hour': hour_str,
                    'entrances': data['entrances'],
                    'avg_dwell': data['avg_counter_dwell'],
                    'interaction_rate': data['counter_interaction_rate'],
                    'danger': 'HIGH'  # Silent leaks are always high danger
                })
        
        if false_calm_hours:
            insight = {
                'pattern': 'false_calm',
                'title': 'SILENT LOSS - OWNER FEELS SAFE',
                'evidence': f"{len(false_calm_hours)} hours showed dangerous false calm",
                'value_loss': ['Revenue', 'Trust', 'Opportunity'],
                'examples': false_calm_hours[:2],
                'confidence': 'HIGH',  # False calm is always high confidence when detected
                'truth_statement': self.patterns['false_calm']['truth'],
                'warning': '⚠ MOST DANGEROUS PATTERN - Value leaks while system appears normal'
            }
            self.insights.append(insight)
            self.value_loss['Revenue'] += len(false_calm_hours) * 2  # Double weight for false calm
            print(f"     🔥 {len(false_calm_hours)} FALSE CALM hours (Most dangerous)")
    
    def _ask_behavioral_questions(self, hourly_data: Dict):
        """Ask the critical behavioral questions."""
        print("   • Asking the hard questions...")
        
        questions_answered = []
        
        # Question 1: Did people come and leave too fast?
        fast_exit_hours = []
        for hour, data in hourly_data.items():
            if data['avg_counter_dwell'] < 45 and data['entrances'] >= 3:
                fast_exit_hours.append(hour.strftime('%H:%M'))
        
        if fast_exit_hours:
            questions_answered.append({
                'question': "Did people come and leave too fast?",
                'answer': f"YES - In hours: {', '.join(fast_exit_hours[:3])}",
                'implication': "Minimal engagement, maximum throughput, minimum value"
            })
        
        # Question 2: Was someone missing when needed?
        unattended_count = len([a for a in self.data['alerts'] 
                              if a.get('alert_type') == 'unattended_station'])
        if unattended_count > 0:
            questions_answered.append({
                'question': "Was someone missing when needed?",
                'answer': f"YES - {unattended_count} unattended alerts",
                'implication': "Presence assumed, attention absent"
            })
        
        # Question 3: Did alerts repeat without correction?
        alert_types = defaultdict(int)
        for alert in self.data['alerts']:
            alert_types[alert.get('alert_type', 'unknown')] += 1
        
        repeated_alerts = [atype for atype, count in alert_types.items() 
                         if count >= 3 and atype != 'unknown']
        
        if repeated_alerts:
            questions_answered.append({
                'question': "Did alerts repeat without correction?",
                'answer': f"YES - Repeated: {', '.join(repeated_alerts)}",
                'implication': "System warnings ignored, patterns normalized"
            })
        
        # Question 4: Was activity scattered instead of focused?
        focused_hours = 0
        scattered_hours = 0
        for hour, data in hourly_data.items():
            if data['zones_used'] >= 2 and data['entrances'] >= 3:
                if data['counter_interaction_rate'] > 0.5:
                    focused_hours += 1
                else:
                    scattered_hours += 1
        
        if scattered_hours > focused_hours:
            questions_answered.append({
                'question': "Was activity scattered instead of focused?",
                'answer': f"YES - {scattered_hours} scattered vs {focused_hours} focused hours",
                'implication': "Energy dispersed, attention fragmented, value diluted"
            })
        
        # Question 5: Did motion happen without outcome?
        motion_without_outcome = []
        for hour, data in hourly_data.items():
            if (data['avg_motion'] > 0.2 and  # Significant motion
                data['counter_interaction_rate'] < 0.3 and  # Low engagement
                data['entrances'] > 0):  # Some presence
                motion_without_outcome.append(hour.strftime('%H:%M'))
        
        if motion_without_outcome:
            questions_answered.append({
                'question': "Did motion happen without outcome?",
                'answer': f"YES - In hours: {', '.join(motion_without_outcome[:3])}",
                'implication': "Activity without achievement, movement without meaning"
            })
        
        self.data['behavioral_questions'] = questions_answered
    
    def generate_truth_report(self):
        """Generate the Behavioral Truth Intelligence Report."""
        
        if not self.insights:
            # Even if no insights, we must still provide truth
            return self._generate_healthy_verification_report()
        
        print(f"\n{'='*80}")
        print("PICAM BEHAVIORAL TRUTH INTELLIGENCE REPORT v2.0")
        print(f"Date: {self.today.strftime('%Y-%m-%d')}")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Data Sources: {len(self.data['detections'])} detections, "
              f"{len(self.data['motion_events'])} motion events, "
              f"{len([z for z in self.data['zone_stats'] if z.get('date') == self.today.strftime('%Y-%m-%d')])} zones")
        print("="*80)
        
        report = []
        
        # 🔴 1. DAILY TRUTH (1-2 paragraphs)
        report.append("🔴 1. DAILY TRUTH")
        report.append("-"*40)
        daily_truth = self._generate_daily_truth()
        report.append(daily_truth)
        report.append("")
        
        # 🟠 2. WHERE VALUE WAS LOST (Ranked)
        report.append("🟠 2. WHERE VALUE WAS LOST")
        report.append("-"*40)
        value_loss_analysis = self._analyze_value_loss()
        for item in value_loss_analysis:
            report.append(item)
        report.append("")
        
        # 🟡 3. WHY THIS HAPPENED (Behavioral Reasoning)
        report.append("🟡 3. WHY THIS HAPPENED")
        report.append("-"*40)
        behavioral_reasons = self._explain_behavioral_reasons()
        for reason in behavioral_reasons:
            report.append(reason)
        report.append("")
        
        # 🟢 4. WHAT TO FIX TOMORROW (Very Practical)
        report.append("🟢 4. WHAT TO FIX TOMORROW")
        report.append("-"*40)
        fixes = self._generate_practical_fixes()
        for fix in fixes:
            report.append(fix)
        report.append("")
        
        # 🔵 5. CONFIDENCE STATEMENT
        report.append("🔵 5. CONFIDENCE STATEMENT")
        report.append("-"*40)
        confidence = self._generate_confidence_statement()
        for line in confidence:
            report.append(line)
        report.append("")
        
        # 🟣 6. THE UNASKED QUESTION ANSWERED
        report.append("🟣 6. THE UNASKED QUESTION ANSWERED")
        report.append("-"*40)
        unasked_answer = self._answer_unasked_question()
        report.append(unasked_answer)
        report.append("")
        
        # FINAL TRUTH VERIFICATION
        report.append("FINAL TRUTH VERIFICATION")
        report.append("-"*40)
        final_truth = self._verify_final_truth()
        report.extend(final_truth)
        report.append("")
        
        report.append("="*80)
        report.append("PICAM BEHAVIORAL TRUTH SYSTEM")
        report.append("Seeing what's there, revealing what's missing")
        report.append("Analysis Complete ✓")
        report.append("="*80)
        
        full_report = "\n".join(report)
        
        # Save report
        self._save_report(full_report)
        
        # Display report
        print(full_report)
        
        return full_report
    
    def _generate_daily_truth(self) -> str:
        """Generate 1-2 paragraph daily truth in plain language."""
        
        if not self.insights:
            return "The system recorded minimal activity today. While no obvious failures were detected, \n" \
                   "the absence of meaningful engagement patterns suggests either very low traffic or \n" \
                   "a disconnect between presence and interaction. Verify human engagement was actually occurring."
        
        # Count pattern types
        pattern_counts = defaultdict(int)
        for insight in self.insights:
            pattern_counts[insight['pattern']] += 1
        
        # Build truth statement
        truth_parts = []
        
        if pattern_counts.get('opportunity_loss'):
            truth_parts.append("Footfall existed but engagement failed.")
        
        if pattern_counts.get('false_calm'):
            truth_parts.append("Silent value leaks occurred while the system appeared normal.")
        
        if pattern_counts.get('responsibility_failure') or pattern_counts.get('chaos_windows'):
            truth_parts.append("Control and attention gaps were evident.")
        
        # Combine into paragraphs
        if truth_parts:
            paragraph1 = " ".join(truth_parts) + " Customers entered and exited without meaningful interaction. " \
                        "This is not congestion — this is neglect."
            
            total_hours = sum([len(self.data.get('behavioral_questions', []))] + 
                            [pattern_counts.get(p, 0) for p in self.patterns.keys()])
            
            paragraph2 = f"Behavioral analysis revealed {total_hours} distinct failure patterns. " \
                        f"The machinery of your operation worked; the humanity was missing. " \
                        f"Value wasn't destroyed — it was never captured."
            
            return f"{paragraph1}\n\n{paragraph2}"
        
        return "Behavioral patterns suggest normal operations, but verify engagement quality manually."
    
    def _analyze_value_loss(self) -> List[str]:
        """Analyze where value was lost, ranked."""
        analysis = []
        
        # Count value losses
        value_loss_items = sorted(self.value_loss.items(), key=lambda x: x[1], reverse=True)
        
        if not value_loss_items:
            analysis.append("No significant value loss detected today.")
            analysis.append("(But verify this isn't False Calm - the most dangerous pattern)")
            return analysis
        
        analysis.append("Ranked by impact:")
        
        for i, (value_type, count) in enumerate(value_loss_items, 1):
            severity = "CRITICAL" if count >= 3 else "HIGH" if count >= 2 else "MEDIUM"
            
            # Add specific examples
            examples = []
            for insight in self.insights:
                if value_type in insight.get('value_loss', []):
                    examples.append(insight.get('title', ''))
            
            example_str = f" - e.g., {examples[0]}" if examples else ""
            
            analysis.append(f"{i}. {value_type} ({severity}){example_str}")
        
        return analysis
    
    def _explain_behavioral_reasons(self) -> List[str]:
        """Explain why this happened using behavioral reasoning."""
        reasons = []
        
        if not self.insights:
            reasons.append("Limited behavioral data available.")
            reasons.append("Either:")
            reasons.append("  • Very low actual activity")
            reasons.append("  • Engagement occurring outside detection patterns")
            reasons.append("  • System needs calibration for your specific environment")
            return reasons
        
        reasons.append("CAUSE → EFFECT → LOSS Analysis:")
        reasons.append("")
        
        for insight in self.insights:
            pattern_info = self.patterns.get(insight['pattern'], {})
            reasons.append(f"• {pattern_info.get('name', insight['pattern'].upper())}:")
            reasons.append(f"  Cause: {pattern_info.get('pattern', 'Unknown pattern')}")
            reasons.append(f"  Effect: {insight.get('truth_statement', 'Behavioral failure')}")
            reasons.append(f"  Loss: {', '.join(insight.get('value_loss', ['Unknown']))}")
            
            if insight.get('examples'):
                example = insight['examples'][0]
                if isinstance(example, dict) and 'hour' in example:
                    reasons.append(f"  Example: {example['hour']} - {insight.get('evidence', '')[:50]}...")
            reasons.append("")
        
        return reasons
    
    def _generate_practical_fixes(self) -> List[str]:
        """Generate very practical fixes for tomorrow."""
        fixes = []
        
        fixes.append("IMMEDIATE ACTION (Tomorrow's Opening):")
        fixes.append("")
        
        # Check for specific patterns and suggest fixes
        pattern_fixes = {
            'opportunity_loss': [
                "1. Counter Presence Protocol:",
                "   • First visitor: Staff MUST engage for minimum 60 seconds",
                "   • Measure: Engagement time, not just occupancy",
                "   • Success: 2+ minute meaningful interactions"
            ],
            'false_calm': [
                "2. Silent Leak Detection:",
                "   • Assign 'Engagement Captain' for first 2 hours",
                "   • Track: Who connected with whom?",
                "   • Alert: Any face >1min without interaction"
            ],
            'responsibility_failure': [
                "3. Zone Ownership:",
                "   • Counter → Named staff (not just 'someone')",
                "   • Entrance → Greeting + direction responsibility",
                "   • Table → 15-minute check-in requirement"
            ],
            'chaos_windows': [
                "4. Control Recovery:",
                "   • Identify peak chaos hour from today's data",
                "   • Pre-assign staff for that hour tomorrow",
                "   • Implement 'Pause & Assess' at chaos onset"
            ]
        }
        
        # Add fixes for detected patterns
        added_fixes = set()
        for insight in self.insights:
            pattern = insight['pattern']
            if pattern in pattern_fixes and pattern not in added_fixes:
                fixes.extend(pattern_fixes[pattern])
                fixes.append("")
                added_fixes.add(pattern)
        
        # Default fixes if no specific patterns
        if not added_fixes:
            fixes.extend([
                "1. Engagement Verification:",
                "   • Staff question: 'Who did you meaningfully engage today?'",
                "   • Not 'Were you busy?' but 'Who connected?'",
                "",
                "2. Value Capture Check:",
                "   • 11 AM and 3 PM: 5-minute engagement audit",
                "   • Count: Meaningful conversations, not transactions",
                ""
            ])
        
        fixes.append("BEHAVIORAL SHIFTS (This Week):")
        fixes.append("• From Motion → Meaning: Celebrate engagement, not activity")
        fixes.append("• From Presence → Partnership: Staff job is connection, not occupancy")
        fixes.append("• From Monitoring → Mentoring: Use alerts for coaching, not punishment")
        
        return fixes
    
    def _generate_confidence_statement(self) -> List[str]:
        """Generate confidence statement."""
        confidence = []
        
        total_detections = len(self.data['detections'])
        total_hours = len(self._create_hourly_behavioral_analysis())
        pattern_count = len(self.insights)
        
        # Calculate confidence score
        if total_detections < 100:
            confidence_level = "LOW"
            reason = "Insufficient behavioral data"
        elif pattern_count >= 3:
            confidence_level = "HIGH"
            reason = "Multiple consistent behavioral patterns"
        elif pattern_count >= 1:
            confidence_level = "MEDIUM"
            reason = "Some behavioral patterns detected"
        else:
            confidence_level = "LOW"
            reason = "No clear behavioral patterns"
        
        confidence.append(f"{confidence_level} CONFIDENCE — {reason}")
        confidence.append("")
        confidence.append("Why:")
        confidence.append(f"• Data Duration: {total_hours} hours analyzed")
        confidence.append(f"• Pattern Repetition: {pattern_count} distinct behavioral patterns")
        confidence.append(f"• Signal Consistency: {total_detections} behavioral markers")
        
        if self.data.get('behavioral_questions'):
            answered = len([q for q in self.data['behavioral_questions'] 
                          if 'YES' in q.get('answer', '')])
            confidence.append(f"• Behavioral Questions: {answered}/5 confirmed issues")
        
        confidence.append("")
        confidence.append("Limitation Note: We detect behavioral patterns, not audio/content.")
        confidence.append("Silent successful engagements may be missed.")
        
        return confidence
    
    def _answer_unasked_question(self) -> str:
        """Answer 'Should I be worried?'"""
        
        if not self.insights:
            return "No — but verify. Your system shows minimal issues, but ensure this isn't " \
                   "'False Calm.' Check manually: Is meaningful engagement actually happening? " \
                   "If yes, your operations are healthy. If no, value is leaking silently."
        
        critical_count = sum(1 for insight in self.insights 
                           if insight.get('pattern') == 'false_calm')
        
        if critical_count > 0:
            return "Yes — but not panicked. You're not losing customers; you're losing " \
                   "THEIR POTENTIAL. Today's data shows silent value leaks. The machinery " \
                   "works; the humanity is missing. This is fixable with intentional engagement."
        else:
            return "Moderately — you have clear behavioral gaps but no silent leaks. " \
                   "Fix the engagement patterns revealed today. Tomorrow's metric: " \
                   "Not 'how busy?' but 'who connected?'"
    
    def _verify_final_truth(self) -> List[str]:
        """Final truth verification."""
        verification = []
        
        pre_assumption = "Operations are running smoothly"
        
        if not self.insights:
            post_truth = "Operations appear normal, but verify engagement quality manually"
        else:
            critical_patterns = [i for i in self.insights 
                               if i.get('pattern') in ['false_calm', 'opportunity_loss']]
            
            if critical_patterns:
                post_truth = "Operations are running, but value is walking out the door"
            else:
                post_truth = "Operations have clear gaps but are fundamentally functional"
        
        verification.append(f"Pre-Report Assumption: '{pre_assumption}'")
        verification.append(f"Post-Report Truth: '{post_truth}'")
        verification.append("")
        verification.append("The evidence is in what's missing:")
        verification.append("• The conversations that didn't happen")
        verification.append("• The relationships that didn't form")
        verification.append("• The trust that didn't grow")
        verification.append("")
        verification.append("Your system recorded activity; your business needs engagement.")
        verification.append("")
        verification.append("Action starts not with technology, but with intention.")
        verification.append("Tomorrow: One meaningful connection per staff member per hour.")
        verification.append("Measure that.")
        
        return verification
    
    def _generate_healthy_verification_report(self):
        """Generate report when no insights found (must verify deeply)."""
        report = []
        
        report.append("="*80)
        report.append("PICAM BEHAVIORAL TRUTH VERIFICATION REQUIRED")
        report.append(f"Date: {self.today.strftime('%Y-%m-%d')}")
        report.append("="*80)
        report.append("")
        report.append("⚠ NO CLEAR BEHAVIORAL PATTERNS DETECTED")
        report.append("")
        report.append("This could mean:")
        report.append("1. ✅ Operations are genuinely healthy and engaged")
        report.append("2. ⚠ 'False Calm' - Silent value leaks are occurring")
        report.append("3. 📊 Insufficient data for pattern detection")
        report.append("")
        report.append("VERIFICATION REQUIRED:")
        report.append("")
        report.append("Before declaring 'healthy', manually verify:")
        report.append("• Are meaningful conversations happening at the counter?")
        report.append("• Is staff engagement intentional, not just presence?")
        report.append("• Are customers leaving satisfied or just processed?")
        report.append("")
        report.append("RECOMMENDED VERIFICATION ACTIONS:")
        report.append("1. Tomorrow 11 AM: 15-minute engagement audit")
        report.append("2. Staff question: 'Who did you connect with today?'")
        report.append("3. Customer spot-check: 'Did you feel attended to?'")
        report.append("")
        report.append("="*80)
        report.append("Truth requires verification, not just data.")
        report.append("="*80)
        
        full_report = "\n".join(report)
        print(full_report)
        self._save_report(full_report, suffix="_verification_required")
        
        return full_report
    
    def _save_report(self, report: str, suffix: str = ""):
        """Save report to file."""
        if suffix:
            filename = f"behavioral_truth_report_{self.today_str}{suffix}.txt"
        else:
            filename = f"behavioral_truth_report_{self.today_str}.txt"
        
        filepath = os.path.join(self.reports_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n✅ Behavioral Truth Report saved to: {filepath}")
            
            # Also save insights as JSON for future analysis
            insights_file = os.path.join(self.reports_dir, f"behavioral_insights_{self.today_str}.json")
            with open(insights_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'date': self.today_str,
                    'insights': self.insights,
                    'value_loss': dict(self.value_loss),
                    'questions': self.data.get('behavioral_questions', [])
                }, f, indent=2, default=str)
            print(f"✅ Behavioral insights saved to: {insights_file}")
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")


def main():
    """Main function for PICAM Behavioral Truth Intelligence."""
    print("\n" + "="*80)
    print("PICAM BEHAVIORAL TRUTH INTELLIGENCE SYSTEM")
    print("="*80)
    print("\nCORE MISSION: Answer ONE question every day:")
    print("'Where did we lose value today — and why?'")
    print("\nValue may be: Revenue, Attention, Trust, Control, Discipline, Opportunity")
    print("\n" + "="*80)
    
    # Initialize system
    truth_system = BehavioralTruthIntelligence()
    
    # Load and analyze data
    if not truth_system.load_and_analyze():
        print("\n❌ Failed to load behavioral data.")
        print("   Ensure you've run the video processor today.")
        return
    
    # Generate truth report
    print(f"\n{'='*80}")
    print("GENERATING BEHAVIORAL TRUTH REPORT")
    print("="*80)
    
    truth_system.generate_truth_report()


if __name__ == "__main__":
    main()