"""
Performance Optimization Module for Visualization Framework
=======================================================

This module provides performance optimizations for the visualization framework,
including lazy plot creation, figure caching, memory management, and
performance benchmarking.

Key Features:
- Lazy plot creation (only when needed)
- Figure caching for repeated plots
- Automatic memory management
- Plot complexity scaling based on data size
- Background plot generation
- Performance benchmarking and monitoring

Performance Focus:
- Reduce memory usage for large datasets
- Improve responsiveness for interactive use
- Optimize plot generation for complex visualizations
- Provide performance insights and recommendations
"""

import time
import weakref
import threading
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import logging
import psutil
import gc
from functools import wraps, lru_cache
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PlotComplexity(Enum):
    """Enumeration for plot complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class PerformanceMetrics:
    """Data class for storing performance metrics."""
    creation_time: float
    memory_usage: float
    figure_size: Tuple[int, int]
    data_points: int
    complexity: PlotComplexity
    cache_hit: bool = False


class PerformanceMonitor:
    """
    Monitor and track performance metrics for visualization operations.
    
    This class provides performance monitoring capabilities including:
    - Timing plot creation operations
    - Memory usage tracking
    - Performance benchmarking
    - Optimization recommendations
    """
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.metrics_history: List[PerformanceMetrics] = []
        self.enabled = True
    
    def start_timer(self) -> float:
        """Start a performance timer."""
        return time.time()
    
    def end_timer(self, start_time: float) -> float:
        """End a performance timer and return elapsed time."""
        return time.time() - start_time
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        if self.enabled:
            self.metrics_history.append(metrics)
            logger.debug(f"Performance metrics recorded: {metrics}")
    
    def get_average_creation_time(self) -> float:
        """Get average plot creation time."""
        if not self.metrics_history:
            return 0.0
        return sum(m.creation_time for m in self.metrics_history) / len(self.metrics_history)
    
    def get_memory_trend(self) -> Dict[str, float]:
        """Get memory usage trend statistics."""
        if not self.metrics_history:
            return {"min": 0.0, "max": 0.0, "avg": 0.0}
        
        memory_values = [m.memory_usage for m in self.metrics_history]
        return {
            "min": min(memory_values),
            "max": max(memory_values),
            "avg": sum(memory_values) / len(memory_values)
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        if not self.metrics_history:
            return {"message": "No performance data available"}
        
        total_plots = len(self.metrics_history)
        cache_hits = sum(1 for m in self.metrics_history if m.cache_hit)
        cache_hit_rate = cache_hits / total_plots if total_plots > 0 else 0
        
        complexity_counts = {}
        for metric in self.metrics_history:
            complexity = metric.complexity.value
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        return {
            "total_plots": total_plots,
            "average_creation_time": self.get_average_creation_time(),
            "memory_trend": self.get_memory_trend(),
            "cache_hit_rate": cache_hit_rate,
            "complexity_distribution": complexity_counts,
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if not self.metrics_history:
            return ["No performance data available for recommendations"]
        
        avg_time = self.get_average_creation_time()
        memory_trend = self.get_memory_trend()
        
        if avg_time > 1.0:
            recommendations.append("Consider using lazy plot creation for slow plots")
        
        if memory_trend["max"] > 500:  # 500MB threshold
            recommendations.append("High memory usage detected - consider figure caching")
        
        cache_hit_rate = sum(1 for m in self.metrics_history if m.cache_hit) / len(self.metrics_history)
        if cache_hit_rate < 0.3:
            recommendations.append("Low cache hit rate - consider enabling more aggressive caching")
        
        return recommendations


class FigureCache:
    """
    Cache for matplotlib figures to improve performance.
    
    This class provides figure caching capabilities including:
    - LRU cache for frequently used figures
    - Memory-aware cache management
    - Automatic cache cleanup
    - Cache hit/miss tracking
    """
    
    def __init__(self, max_size: int = 50, max_memory_mb: float = 100.0):
        """
        Initialize the figure cache.
        
        Args:
            max_size: Maximum number of cached figures
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.cache: Dict[str, Tuple[Figure, float]] = {}
        self.access_times: Dict[str, float] = {}
        self.monitor = PerformanceMonitor()
    
    def _generate_cache_key(self, plot_type: str, data_hash: str, **kwargs) -> str:
        """Generate a unique cache key for a plot."""
        # Create a hash of the kwargs for cache key
        kwargs_str = str(sorted(kwargs.items()))
        return f"{plot_type}_{data_hash}_{hash(kwargs_str)}"
    
    def _get_figure_size(self, fig: Figure) -> float:
        """Estimate figure memory usage in MB."""
        # Rough estimation: 1MB per 1000x1000 pixels
        width, height = fig.get_size_inches()
        dpi = fig.dpi
        pixels = width * height * dpi * dpi
        return pixels / 1000000  # Convert to MB
    
    def get(self, plot_type: str, data_hash: str, **kwargs) -> Optional[Figure]:
        """
        Get a cached figure if available.
        
        Args:
            plot_type: Type of plot
            data_hash: Hash of the data
            **kwargs: Plot parameters
            
        Returns:
            Cached figure or None if not found
        """
        cache_key = self._generate_cache_key(plot_type, data_hash, **kwargs)
        
        if cache_key in self.cache:
            fig, _ = self.cache[cache_key]
            self.access_times[cache_key] = time.time()
            
            # Record cache hit
            metrics = PerformanceMetrics(
                creation_time=0.0,  # Cache hit, no creation time
                memory_usage=self.monitor.get_memory_usage(),
                figure_size=fig.get_size_inches(),
                data_points=0,  # Unknown for cached figures
                complexity=PlotComplexity.MEDIUM,
                cache_hit=True
            )
            self.monitor.record_metrics(metrics)
            
            logger.debug(f"Cache hit for {plot_type}")
            return fig
        
        logger.debug(f"Cache miss for {plot_type}")
        return None
    
    def put(self, plot_type: str, data_hash: str, fig: Figure, **kwargs) -> None:
        """
        Store a figure in the cache.
        
        Args:
            plot_type: Type of plot
            data_hash: Hash of the data
            fig: Figure to cache
            **kwargs: Plot parameters
        """
        cache_key = self._generate_cache_key(plot_type, data_hash, **kwargs)
        
        # Check if we need to evict items
        self._evict_if_needed()
        
        # Store the figure
        self.cache[cache_key] = (fig, time.time())
        self.access_times[cache_key] = time.time()
        
        logger.debug(f"Cached figure for {plot_type}")
    
    def _evict_if_needed(self) -> None:
        """Evict cache items if necessary."""
        # Check size limit
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        # Check memory limit
        total_memory = sum(self._get_figure_size(fig) for fig, _ in self.cache.values())
        if total_memory > self.max_memory_mb:
            self._evict_largest()
    
    def _evict_lru(self) -> None:
        """Evict least recently used items."""
        if not self.access_times:
            return
        
        # Find the least recently used item
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove from cache
        if lru_key in self.cache:
            fig, _ = self.cache.pop(lru_key)
            plt.close(fig)  # Close the figure to free memory
        
        # Remove from access times
        self.access_times.pop(lru_key, None)
        
        logger.debug(f"Evicted LRU figure: {lru_key}")
    
    def _evict_largest(self) -> None:
        """Evict largest figures to free memory."""
        if not self.cache:
            return
        
        # Find the largest figure
        largest_key = max(
            self.cache.keys(),
            key=lambda k: self._get_figure_size(self.cache[k][0])
        )
        
        # Remove from cache
        if largest_key in self.cache:
            fig, _ = self.cache.pop(largest_key)
            plt.close(fig)  # Close the figure to free memory
        
        # Remove from access times
        self.access_times.pop(largest_key, None)
        
        logger.debug(f"Evicted largest figure: {largest_key}")
    
    def clear(self) -> None:
        """Clear all cached figures."""
        for fig, _ in self.cache.values():
            plt.close(fig)
        
        self.cache.clear()
        self.access_times.clear()
        
        logger.debug("Figure cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_memory = sum(self._get_figure_size(fig) for fig, _ in self.cache.values())
        
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "total_memory_mb": total_memory,
            "max_memory_mb": self.max_memory_mb,
            "memory_usage_percent": (total_memory / self.max_memory_mb) * 100 if self.max_memory_mb > 0 else 0
        }


class LazyPlotCreator:
    """
    Lazy plot creation for improved performance.
    
    This class provides lazy plot creation capabilities including:
    - Deferred plot creation until needed
    - Background plot generation
    - Plot complexity estimation
    - Performance-aware plot scaling
    """
    
    def __init__(self, cache: Optional[FigureCache] = None):
        """
        Initialize the lazy plot creator.
        
        Args:
            cache: Optional figure cache for reuse
        """
        self.cache = cache or FigureCache()
        self.monitor = PerformanceMonitor()
        self._background_threads: List[threading.Thread] = []
    
    def estimate_complexity(self, data_size: int, plot_type: str) -> PlotComplexity:
        """
        Estimate plot complexity based on data size and type.
        
        Args:
            data_size: Number of data points
            plot_type: Type of plot
            
        Returns:
            Estimated complexity level
        """
        if data_size < 100:
            return PlotComplexity.SIMPLE
        elif data_size < 1000:
            return PlotComplexity.MEDIUM
        elif data_size < 10000:
            return PlotComplexity.COMPLEX
        else:
            return PlotComplexity.VERY_COMPLEX
    
    def create_plot_lazy(self, 
                        plot_func: Callable,
                        data_hash: str,
                        plot_type: str,
                        data_size: int,
                        **kwargs) -> Figure:
        """
        Create a plot lazily with performance optimization.
        
        Args:
            plot_func: Function to create the plot
            data_hash: Hash of the data for caching
            plot_type: Type of plot
            data_size: Number of data points
            **kwargs: Arguments for plot function
            
        Returns:
            Created figure
        """
        start_time = self.monitor.start_timer()
        
        # Check cache first
        cached_fig = self.cache.get(plot_type, data_hash, **kwargs)
        if cached_fig is not None:
            return cached_fig
        
        # Estimate complexity
        complexity = self.estimate_complexity(data_size, plot_type)
        
        # Create the plot
        fig = plot_func(**kwargs)
        
        # Record metrics
        creation_time = self.monitor.end_timer(start_time)
        memory_usage = self.monitor.get_memory_usage()
        
        metrics = PerformanceMetrics(
            creation_time=creation_time,
            memory_usage=memory_usage,
            figure_size=fig.get_size_inches(),
            data_points=data_size,
            complexity=complexity,
            cache_hit=False
        )
        self.monitor.record_metrics(metrics)
        
        # Cache the figure
        self.cache.put(plot_type, data_hash, fig, **kwargs)
        
        logger.debug(f"Created {plot_type} plot in {creation_time:.3f}s")
        return fig
    
    def create_plot_background(self,
                             plot_func: Callable,
                             data_hash: str,
                             plot_type: str,
                             data_size: int,
                             callback: Optional[Callable[[Figure], None]] = None,
                             **kwargs) -> None:
        """
        Create a plot in the background.
        
        Args:
            plot_func: Function to create the plot
            data_hash: Hash of the data for caching
            plot_type: Type of plot
            data_size: Number of data points
            callback: Optional callback when plot is ready
            **kwargs: Arguments for plot function
        """
        def _background_creation():
            try:
                fig = self.create_plot_lazy(plot_func, data_hash, plot_type, data_size, **kwargs)
                if callback:
                    callback(fig)
            except Exception as e:
                logger.error(f"Background plot creation failed: {e}")
        
        thread = threading.Thread(target=_background_creation)
        thread.daemon = True
        thread.start()
        
        self._background_threads.append(thread)
        
        logger.debug(f"Started background creation of {plot_type} plot")
    
    def cleanup_background_threads(self) -> None:
        """Clean up completed background threads."""
        self._background_threads = [t for t in self._background_threads if t.is_alive()]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report."""
        return self.monitor.get_performance_report()


class MemoryManager:
    """
    Memory management for visualization operations.
    
    This class provides memory management capabilities including:
    - Automatic garbage collection
    - Memory usage monitoring
    - Figure cleanup
    - Memory optimization recommendations
    """
    
    def __init__(self, threshold_mb: float = 100.0):
        """
        Initialize the memory manager.
        
        Args:
            threshold_mb: Memory threshold in MB for cleanup
        """
        self.threshold_mb = threshold_mb
        self.monitor = PerformanceMonitor()
    
    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage."""
        current_memory = self.monitor.get_memory_usage()
        return {
            "current_mb": current_memory,
            "threshold_mb": self.threshold_mb,
            "usage_percent": (current_memory / self.threshold_mb) * 100 if self.threshold_mb > 0 else 0
        }
    
    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed."""
        memory_info = self.check_memory_usage()
        return memory_info["current_mb"] > self.threshold_mb
    
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """
        Perform memory cleanup.
        
        Args:
            aggressive: Whether to perform aggressive cleanup
            
        Returns:
            Cleanup results
        """
        initial_memory = self.monitor.get_memory_usage()
        
        # Force garbage collection
        gc.collect()
        
        # Close unused matplotlib figures
        plt.close('all')
        
        if aggressive:
            # More aggressive cleanup
            gc.collect()
            gc.collect()  # Double collection for aggressive cleanup
        
        final_memory = self.monitor.get_memory_usage()
        memory_freed = initial_memory - final_memory
        
        result = {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_freed_mb": memory_freed,
            "cleanup_type": "aggressive" if aggressive else "normal"
        }
        
        logger.info(f"Memory cleanup completed: {memory_freed:.2f}MB freed")
        return result
    
    def optimize_for_data_size(self, data_size: int) -> Dict[str, Any]:
        """
        Optimize memory settings for data size.
        
        Args:
            data_size: Number of data points
            
        Returns:
            Optimization recommendations
        """
        recommendations = []
        
        if data_size > 10000:
            recommendations.append("Consider using data sampling for large datasets")
            recommendations.append("Enable aggressive memory cleanup")
        elif data_size > 1000:
            recommendations.append("Consider using figure caching")
            recommendations.append("Monitor memory usage closely")
        else:
            recommendations.append("Standard memory settings are appropriate")
        
        return {
            "data_size": data_size,
            "recommendations": recommendations,
            "suggested_cache_size": min(50, max(10, data_size // 100))
        }


# Performance decorators for easy integration
def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = PerformanceMonitor()
        start_time = monitor.start_timer()
        
        try:
            result = func(*args, **kwargs)
            
            # Record metrics
            creation_time = monitor.end_timer(start_time)
            memory_usage = monitor.get_memory_usage()
            
            # Estimate data size and complexity
            data_size = 0
            if args and hasattr(args[0], 'shape'):
                data_size = args[0].shape[0]
            elif 'data' in kwargs and hasattr(kwargs['data'], 'shape'):
                data_size = kwargs['data'].shape[0]
            
            complexity = PlotComplexity.MEDIUM  # Default
            if data_size > 0:
                if data_size < 100:
                    complexity = PlotComplexity.SIMPLE
                elif data_size < 1000:
                    complexity = PlotComplexity.MEDIUM
                elif data_size < 10000:
                    complexity = PlotComplexity.COMPLEX
                else:
                    complexity = PlotComplexity.VERY_COMPLEX
            
            metrics = PerformanceMetrics(
                creation_time=creation_time,
                memory_usage=memory_usage,
                figure_size=(8, 6),  # Default size
                data_points=data_size,
                complexity=complexity
            )
            monitor.record_metrics(metrics)
            
            return result
        except Exception as e:
            logger.error(f"Performance monitoring failed for {func.__name__}: {e}")
            raise
    
    return wrapper


def lazy_plot_creation(cache: Optional[FigureCache] = None):
    """Decorator for lazy plot creation."""
    def decorator(func: Callable) -> Callable:
        lazy_creator = LazyPlotCreator(cache)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate data hash for caching
            import hashlib
            data_str = str(args) + str(sorted(kwargs.items()))
            data_hash = hashlib.md5(data_str.encode()).hexdigest()
            
            # Estimate data size
            data_size = 0
            if args and hasattr(args[0], 'shape'):
                data_size = args[0].shape[0]
            elif 'data' in kwargs and hasattr(kwargs['data'], 'shape'):
                data_size = kwargs['data'].shape[0]
            
            plot_type = func.__name__
            
            return lazy_creator.create_plot_lazy(
                lambda **kw: func(*args, **kw),
                data_hash,
                plot_type,
                data_size,
                **kwargs
            )
        
        return wrapper
    return decorator


# Convenience functions
def get_performance_summary() -> Dict[str, Any]:
    """Get a summary of current performance metrics."""
    monitor = PerformanceMonitor()
    return monitor.get_performance_report()


def optimize_memory_for_large_data(data_size: int) -> Dict[str, Any]:
    """Optimize memory settings for large datasets."""
    manager = MemoryManager()
    return manager.optimize_for_data_size(data_size)


def cleanup_memory_if_needed() -> Dict[str, Any]:
    """Clean up memory if usage is high."""
    manager = MemoryManager()
    if manager.should_cleanup():
        return manager.cleanup_memory()
    return {"message": "Memory usage is within acceptable limits"} 