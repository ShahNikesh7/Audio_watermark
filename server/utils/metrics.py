"""
Metrics collection and reporting for SoundSafeAI.
"""

import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    count: int
    sum: float
    min: float
    max: float
    mean: float
    
    @classmethod
    def from_values(cls, values: List[float]) -> 'MetricSummary':
        """Create summary from list of values."""
        if not values:
            return cls(0, 0.0, 0.0, 0.0, 0.0)
        
        return cls(
            count=len(values),
            sum=sum(values),
            min=min(values),
            max=max(values),
            mean=sum(values) / len(values)
        )


class MetricsCollector:
    """Collects and aggregates metrics for SoundSafeAI."""
    
    def __init__(self, max_points: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            max_points: Maximum number of metric points to keep in memory
        """
        self.max_points = max_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # Service start time
        self.service_start_time = time.time()
    
    def counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """
        Increment a counter metric.
        
        Args:
            name: Metric name
            value: Value to add to counter
            tags: Optional tags for the metric
        """
        with self._lock:
            key = self._get_metric_key(name, tags)
            self.counters[key] += value
            
            # Also store as time series
            self.metrics[key].append(MetricPoint(
                timestamp=time.time(),
                value=self.counters[key],
                tags=tags or {}
            ))
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Set a gauge metric.
        
        Args:
            name: Metric name
            value: Current value
            tags: Optional tags for the metric
        """
        with self._lock:
            key = self._get_metric_key(name, tags)
            self.gauges[key] = value
            
            # Also store as time series
            self.metrics[key].append(MetricPoint(
                timestamp=time.time(),
                value=value,
                tags=tags or {}
            ))
    
    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Record a histogram metric.
        
        Args:
            name: Metric name
            value: Value to record
            tags: Optional tags for the metric
        """
        with self._lock:
            key = self._get_metric_key(name, tags)
            self.histograms[key].append(value)
            
            # Also store as time series
            self.metrics[key].append(MetricPoint(
                timestamp=time.time(),
                value=value,
                tags=tags or {}
            ))
    
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """
        Create a timer context manager.
        
        Args:
            name: Metric name
            tags: Optional tags for the metric
            
        Returns:
            Timer context manager
        """
        return TimerContext(self, name, tags)
    
    def get_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        key = self._get_metric_key(name, tags)
        return self.counters.get(key, 0.0)
    
    def get_gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        key = self._get_metric_key(name, tags)
        return self.gauges.get(key, 0.0)
    
    def get_histogram_summary(self, name: str, tags: Optional[Dict[str, str]] = None) -> MetricSummary:
        """Get histogram summary statistics."""
        key = self._get_metric_key(name, tags)
        values = self.histograms.get(key, [])
        return MetricSummary.from_values(values)
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get service-level metrics."""
        uptime = time.time() - self.service_start_time
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.get_counter("requests_total"),
            "total_embeds": self.get_counter("embeds_total"),
            "total_extracts": self.get_counter("extracts_total"),
            "total_detections": self.get_counter("detections_total"),
            "average_processing_time": self.get_histogram_summary("processing_time_seconds").mean,
            "error_rate": self.get_counter("errors_total") / max(self.get_counter("requests_total"), 1),
            "active_connections": self.get_gauge("active_connections"),
            "memory_usage_bytes": self.get_gauge("memory_usage_bytes")
        }
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Counters
        for key, value in self.counters.items():
            metric_name, tags = self._parse_metric_key(key)
            tags_str = self._format_prometheus_tags(tags)
            lines.append(f"{metric_name}{tags_str} {value}")
        
        # Gauges
        for key, value in self.gauges.items():
            metric_name, tags = self._parse_metric_key(key)
            tags_str = self._format_prometheus_tags(tags)
            lines.append(f"{metric_name}{tags_str} {value}")
        
        # Histograms (simplified)
        for key, values in self.histograms.items():
            if not values:
                continue
                
            metric_name, tags = self._parse_metric_key(key)
            summary = MetricSummary.from_values(values)
            
            tags_str = self._format_prometheus_tags(tags)
            lines.append(f"{metric_name}_count{tags_str} {summary.count}")
            lines.append(f"{metric_name}_sum{tags_str} {summary.sum}")
        
        return "\n".join(lines)
    
    def _get_metric_key(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Generate metric key from name and tags."""
        if not tags:
            return name
        
        # Sort tags for consistent key generation
        sorted_tags = sorted(tags.items())
        tags_str = ",".join(f"{k}={v}" for k, v in sorted_tags)
        return f"{name}|{tags_str}"
    
    def _parse_metric_key(self, key: str) -> tuple:
        """Parse metric key back to name and tags."""
        if "|" not in key:
            return key, {}
        
        name, tags_str = key.split("|", 1)
        tags = {}
        
        if tags_str:
            for tag_pair in tags_str.split(","):
                k, v = tag_pair.split("=", 1)
                tags[k] = v
        
        return name, tags
    
    def _format_prometheus_tags(self, tags: Dict[str, str]) -> str:
        """Format tags for Prometheus output."""
        if not tags:
            return ""
        
        formatted_tags = [f'{k}="{v}"' for k, v in sorted(tags.items())]
        return "{" + ",".join(formatted_tags) + "}"


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.histogram(self.name, duration, self.tags)


# Global metrics collector instance
metrics = MetricsCollector()


# Convenience functions
def counter(name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
    """Record a counter metric."""
    metrics.counter(name, value, tags)


def gauge(name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Record a gauge metric."""
    metrics.gauge(name, value, tags)


def histogram(name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Record a histogram metric."""
    metrics.histogram(name, value, tags)


def timer(name: str, tags: Optional[Dict[str, str]] = None):
    """Create a timer context manager."""
    return metrics.timer(name, tags)
