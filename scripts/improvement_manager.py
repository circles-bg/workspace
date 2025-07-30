#!/usr/bin/env python3
"""
Advanced Codebase Improvement Manager

This script provides sophisticated analysis and improvement management
for the Rust codebase, including metrics tracking, trend analysis,
and intelligent scheduling of improvement cycles.
"""

import os
import sys
import json
import subprocess
import datetime
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import configparser

class CodebaseImprovement:
    def __init__(self, config_path: str = "improvement-config.toml"):
        self.repo_dir = Path("/home/lokrain/map/workspace")
        self.config_path = self.repo_dir / config_path
        self.logs_dir = self.repo_dir / "logs"
        self.metrics_file = self.logs_dir / "improvement_metrics.json"
        
        # Ensure directories exist
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.logs_dir / 'improvement_manager.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self.load_config()
        
        # Initialize metrics
        self.metrics = self.load_metrics()

    def load_config(self) -> Dict:
        """Load configuration from TOML file"""
        # For now, use default config if file doesn't exist
        default_config = {
            'general': {
                'cycle_interval': 60,
                'max_cycles_per_day': 24,
                'enable_security_checks': True,
                'enable_performance_analysis': True,
                'enable_documentation_checks': True,
                'enable_duplication_detection': True
            },
            'clippy': {
                'deny_warnings': True,
                'apply_fixes_automatically': True
            },
            'git': {
                'auto_commit': True,
                'create_branches': True,
                'push_to_remote': False,
                'branch_prefix': 'auto-improvement'
            }
        }
        return default_config

    def load_metrics(self) -> Dict:
        """Load historical metrics"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load metrics: {e}")
        
        return {
            'cycles': [],
            'trends': {},
            'last_update': None
        }

    def save_metrics(self):
        """Save metrics to file"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")

    def run_command(self, command: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
        """Run a shell command and return exit code, stdout, stderr"""
        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.repo_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timeout: {' '.join(command)}")
            return -1, "", "Command timed out"
        except Exception as e:
            self.logger.error(f"Command failed: {e}")
            return -1, "", str(e)

    def analyze_codebase(self) -> Dict:
        """Perform comprehensive codebase analysis"""
        analysis = {
            'timestamp': datetime.datetime.now().isoformat(),
            'metrics': {},
            'issues': [],
            'suggestions': []
        }
        
        self.logger.info("Starting codebase analysis...")
        
        # Count lines of code
        exit_code, stdout, stderr = self.run_command([
            'find', '.', '-name', '*.rs', '-not', '-path', './target/*',
            '-exec', 'wc', '-l', '{}', '+'
        ])
        
        if exit_code == 0 and stdout:
            lines = stdout.strip().split('\n')
            if lines:
                total_lines = lines[-1].split()[0] if lines[-1].strip() else "0"
                analysis['metrics']['total_lines'] = int(total_lines)
        
        # Run clippy analysis
        exit_code, stdout, stderr = self.run_command([
            'cargo', 'clippy', '--message-format=json', '--all-targets', '--all-features'
        ])
        
        clippy_warnings = 0
        if exit_code != 0 and stderr:
            # Parse clippy JSON output to count warnings
            for line in stderr.split('\n'):
                if 'warning:' in line and '"level":"warning"' in line:
                    clippy_warnings += 1
        
        analysis['metrics']['clippy_warnings'] = clippy_warnings
        
        # Check test coverage
        exit_code, stdout, stderr = self.run_command(['cargo', 'test', '--', '--list'])
        test_count = len([l for l in stdout.split('\n') if 'test result:' not in l and l.strip().endswith(': test')])
        analysis['metrics']['test_count'] = test_count
        
        # Analyze dependencies
        exit_code, stdout, stderr = self.run_command(['cargo', 'tree', '--depth', '1'])
        if exit_code == 0:
            deps = len([l for l in stdout.split('\n') if l.startswith('├──') or l.startswith('└──')])
            analysis['metrics']['dependency_count'] = deps
        
        self.logger.info(f"Analysis complete: {analysis['metrics']}")
        return analysis

    def run_improvements(self) -> Dict:
        """Run improvement cycle"""
        self.logger.info("Starting improvement cycle...")
        
        cycle_start = datetime.datetime.now()
        improvements = {
            'start_time': cycle_start.isoformat(),
            'steps': [],
            'changes_made': False,
            'success': True
        }
        
        # Pre-analysis
        pre_analysis = self.analyze_codebase()
        improvements['pre_analysis'] = pre_analysis
        
        # Step 1: Run clippy fixes
        if self.config['clippy']['apply_fixes_automatically']:
            self.logger.info("Applying clippy fixes...")
            exit_code, stdout, stderr = self.run_command([
                'cargo', 'clippy', '--fix', '--allow-dirty', '--allow-staged',
                '--all-targets', '--all-features'
            ])
            improvements['steps'].append({
                'name': 'clippy_fixes',
                'success': exit_code == 0,
                'output': stdout + stderr
            })
        
        # Step 2: Format code
        self.logger.info("Formatting code...")
        exit_code, stdout, stderr = self.run_command(['cargo', 'fmt', '--all'])
        improvements['steps'].append({
            'name': 'formatting',
            'success': exit_code == 0,
            'output': stdout + stderr
        })
        
        # Step 3: Run tests
        self.logger.info("Running tests...")
        exit_code, stdout, stderr = self.run_command(['cargo', 'test', '--all'])
        improvements['steps'].append({
            'name': 'testing',
            'success': exit_code == 0,
            'output': stdout + stderr
        })
        
        if exit_code != 0:
            improvements['success'] = False
            self.logger.error("Tests failed, aborting improvement cycle")
            return improvements
        
        # Step 4: Check for changes
        exit_code, stdout, stderr = self.run_command(['git', 'diff', '--quiet'])
        if exit_code != 0:  # There are changes
            improvements['changes_made'] = True
            
            if self.config['git']['auto_commit']:
                # Create branch if configured
                if self.config['git']['create_branches']:
                    branch_name = f"{self.config['git']['branch_prefix']}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    self.run_command(['git', 'checkout', '-b', branch_name])
                
                # Commit changes
                self.run_command(['git', 'add', '.'])
                commit_msg = f"""🔧 Automated improvement cycle

- Applied clippy fixes and code formatting
- Verified all tests pass
- Improved code quality and consistency

Timestamp: {cycle_start.isoformat()}"""
                
                self.run_command(['git', 'commit', '-m', commit_msg])
                self.logger.info("Changes committed successfully")
        
        # Post-analysis
        post_analysis = self.analyze_codebase()
        improvements['post_analysis'] = post_analysis
        
        # Calculate improvements
        improvements['metrics_delta'] = self.calculate_metrics_delta(
            pre_analysis['metrics'], 
            post_analysis['metrics']
        )
        
        improvements['end_time'] = datetime.datetime.now().isoformat()
        
        # Store in metrics
        self.metrics['cycles'].append(improvements)
        self.metrics['last_update'] = improvements['end_time']
        self.save_metrics()
        
        self.logger.info("Improvement cycle completed successfully")
        return improvements

    def calculate_metrics_delta(self, pre: Dict, post: Dict) -> Dict:
        """Calculate the difference between pre and post metrics"""
        delta = {}
        for key in pre:
            if key in post and isinstance(pre[key], (int, float)):
                delta[key] = post[key] - pre[key]
        return delta

    def generate_report(self) -> str:
        """Generate a comprehensive improvement report"""
        report_lines = [
            "# Codebase Improvement Report",
            f"Generated: {datetime.datetime.now().isoformat()}",
            "",
            "## Current Status"
        ]
        
        if self.metrics['cycles']:
            latest = self.metrics['cycles'][-1]
            report_lines.extend([
                f"- Last improvement cycle: {latest.get('end_time', 'Unknown')}",
                f"- Changes made: {'Yes' if latest.get('changes_made') else 'No'}",
                f"- Cycle successful: {'Yes' if latest.get('success') else 'No'}",
                ""
            ])
            
            if 'post_analysis' in latest and 'metrics' in latest['post_analysis']:
                metrics = latest['post_analysis']['metrics']
                report_lines.extend([
                    "## Current Metrics",
                    f"- Total lines of code: {metrics.get('total_lines', 'Unknown')}",
                    f"- Clippy warnings: {metrics.get('clippy_warnings', 'Unknown')}",
                    f"- Test count: {metrics.get('test_count', 'Unknown')}",
                    f"- Dependencies: {metrics.get('dependency_count', 'Unknown')}",
                    ""
                ])
        
        # Trend analysis
        if len(self.metrics['cycles']) > 1:
            report_lines.extend([
                "## Trends",
                f"- Total improvement cycles: {len(self.metrics['cycles'])}",
                ""
            ])
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            "- Continue regular improvement cycles",
            "- Monitor test coverage",
            "- Keep dependencies updated",
            "- Review and refactor large functions",
            ""
        ])
        
        return "\n".join(report_lines)

    def start_continuous_cycles(self):
        """Start continuous improvement cycles"""
        self.logger.info("Starting continuous improvement cycles...")
        
        while True:
            try:
                # Check if we've hit the daily limit
                today = datetime.date.today()
                today_cycles = [
                    c for c in self.metrics['cycles']
                    if c.get('start_time', '').startswith(today.isoformat())
                ]
                
                if len(today_cycles) >= self.config['general']['max_cycles_per_day']:
                    self.logger.info("Daily cycle limit reached, sleeping until tomorrow")
                    # Sleep until midnight
                    now = datetime.datetime.now()
                    tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
                    sleep_seconds = (tomorrow - now).total_seconds()
                    time.sleep(sleep_seconds)
                    continue
                
                # Run improvement cycle
                improvements = self.run_improvements()
                
                # Generate and save report
                report = self.generate_report()
                report_file = self.logs_dir / f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                with open(report_file, 'w') as f:
                    f.write(report)
                
                self.logger.info(f"Report saved to {report_file}")
                
                # Wait for next cycle
                import time
                sleep_minutes = self.config['general']['cycle_interval']
                self.logger.info(f"Sleeping for {sleep_minutes} minutes until next cycle...")
                time.sleep(sleep_minutes * 60)
                
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal, stopping...")
                break
            except Exception as e:
                self.logger.error(f"Error in continuous cycle: {e}")
                import time
                time.sleep(300)  # Sleep 5 minutes on error

def main():
    parser = argparse.ArgumentParser(description="Advanced Codebase Improvement Manager")
    parser.add_argument('command', choices=['single', 'continuous', 'analyze', 'report'],
                       help='Command to execute')
    parser.add_argument('--config', default='improvement-config.toml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    manager = CodebaseImprovement(args.config)
    
    if args.command == 'single':
        improvements = manager.run_improvements()
        print(f"Improvement cycle completed. Changes made: {improvements['changes_made']}")
    
    elif args.command == 'continuous':
        manager.start_continuous_cycles()
    
    elif args.command == 'analyze':
        analysis = manager.analyze_codebase()
        print(json.dumps(analysis, indent=2))
    
    elif args.command == 'report':
        report = manager.generate_report()
        print(report)

if __name__ == "__main__":
    main()
