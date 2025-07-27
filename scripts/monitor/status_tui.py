#!/usr/bin/env python3
"""
XORB System Status TUI - HTOP-style monitoring for XORB deployment
"""

import curses
import time
import subprocess
import json
import psutil
import docker
from datetime import datetime

class XorbStatusTUI:
    def __init__(self):
        self.client = docker.from_env()
        self.refresh_interval = 2
        
    def get_container_stats(self):
        containers = []
        try:
            for container in self.client.containers.list():
                if 'xorb' in container.name:
                    stats = container.stats(stream=False)
                    cpu_percent = self.calculate_cpu_percent(stats)
                    memory_usage = stats['memory_stats'].get('usage', 0)
                    memory_limit = stats['memory_stats'].get('limit', 0)
                    memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0
                    
                    containers.append({
                        'name': container.name,
                        'status': container.status,
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory_percent,
                        'memory_usage': memory_usage,
                        'ports': container.ports
                    })
        except Exception as e:
            pass
        return containers
    
    def calculate_cpu_percent(self, stats):
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            if system_delta > 0:
                return (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100
        except (KeyError, ZeroDivisionError):
            pass
        return 0.0
    
    def draw_header(self, stdscr, y=0):
        stdscr.addstr(y, 0, "XORB AUTONOMOUS SECURITY PLATFORM - SYSTEM STATUS", curses.A_BOLD | curses.color_pair(1))
        stdscr.addstr(y+1, 0, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return y + 3
    
    def draw_system_info(self, stdscr, y):
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        stdscr.addstr(y, 0, "SYSTEM RESOURCES:", curses.A_BOLD)
        stdscr.addstr(y+1, 2, f"CPU Usage:    {cpu_percent:6.1f}%")
        stdscr.addstr(y+2, 2, f"Memory Usage: {memory.percent:6.1f}% ({memory.used//1024//1024}MB / {memory.total//1024//1024}MB)")
        stdscr.addstr(y+3, 2, f"Disk Usage:   {disk.percent:6.1f}% ({disk.used//1024//1024//1024}GB / {disk.total//1024//1024//1024}GB)")
        return y + 5
    
    def draw_containers(self, stdscr, y):
        containers = self.get_container_stats()
        
        stdscr.addstr(y, 0, "XORB CONTAINERS:", curses.A_BOLD)
        
        # Header
        stdscr.addstr(y+1, 2, "NAME".ljust(20) + "STATUS".ljust(12) + "CPU%".ljust(8) + "MEM%".ljust(8) + "MEMORY".ljust(12))
        stdscr.addstr(y+2, 2, "-" * 70)
        
        for i, container in enumerate(containers):
            row = y + 3 + i
            if row >= curses.LINES - 2:
                break
                
            status_color = curses.color_pair(2) if container['status'] == 'running' else curses.color_pair(3)
            
            line = f"{container['name'][:19].ljust(20)}"
            line += f"{container['status'].ljust(12)}"
            line += f"{container['cpu_percent']:6.1f}%".ljust(8)
            line += f"{container['memory_percent']:6.1f}%".ljust(8)
            line += f"{container['memory_usage']//1024//1024}MB".ljust(12)
            
            stdscr.addstr(row, 2, line, status_color)
        
        return y + 4 + len(containers)
    
    def draw_logs(self, stdscr, y):
        stdscr.addstr(y, 0, "RECENT LOGS:", curses.A_BOLD)
        
        try:
            # Get recent logs from journalctl
            result = subprocess.run(['journalctl', '-u', 'xorb-*', '--no-pager', '-n', '5', '--output=short'], 
                                  capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            
            for i, line in enumerate(lines[-5:]):
                if y + 1 + i >= curses.LINES - 1:
                    break
                stdscr.addstr(y + 1 + i, 2, line[:curses.COLS-3])
        except:
            stdscr.addstr(y + 1, 2, "No logs available")
        
        return y + 7
    
    def run(self, stdscr):
        curses.curs_set(0)  # Hide cursor
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        
        stdscr.timeout(1000)  # 1 second timeout for getch
        
        while True:
            stdscr.clear()
            
            y = self.draw_header(stdscr)
            y = self.draw_system_info(stdscr, y)
            y = self.draw_containers(stdscr, y)
            y = self.draw_logs(stdscr, y)
            
            stdscr.addstr(curses.LINES-1, 0, "Press 'q' to quit, 'r' to refresh", curses.A_REVERSE)
            stdscr.refresh()
            
            key = stdscr.getch()
            if key == ord('q'):
                break
            elif key == ord('r'):
                continue

if __name__ == "__main__":
    try:
        monitor = XorbStatusTUI()
        curses.wrapper(monitor.run)
    except KeyboardInterrupt:
        print("Monitoring stopped.")
