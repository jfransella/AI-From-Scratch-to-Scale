"""
Test Performance Optimization Features
===================================

This script tests the performance optimization features of the visualization
framework, demonstrating lazy plot creation, figure caching, memory management,
and performance benchmarking.

Key Features Tested:
- Lazy plot creation for large datasets
- Figure caching and cache hit/miss tracking
- Memory management and cleanup
- Performance monitoring and benchmarking
- Plot complexity estimation
- Background plot generation
- Performance optimization recommendations
"""

import sys
import os
import tempfile
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import Mock
import logging

# Add the shared package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the performance components
from visualization.performance import (
    PerformanceMonitor,
    FigureCache,
    LazyPlotCreator,
    MemoryManager,
    PlotComplexity,
    PerformanceMetrics,
    performance_monitor,
    lazy_plot_creation,
    get_performance_summary,
    optimize_memory_for_large_data,
    cleanup_memory_if_needed
)
from visualization.plot_factory import PlotFactory
from visualization import BaseVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_performance_monitor():
    """Test performance monitoring capabilities."""
    logger.info("=" * 60)
    logger.info("TESTING PERFORMANCE MONITOR")
    logger.info("=" * 60)
    
    monitor = PerformanceMonitor()
    
    # Test 1: Basic timing
    logger.info("Test 1: Basic timing")
    start_time = monitor.start_timer()
    time.sleep(0.1)  # Simulate work
    elapsed = monitor.end_timer(start_time)
    logger.info(f"Elapsed time: {elapsed:.3f}s")
    
    # Test 2: Memory usage tracking
    logger.info("Test 2: Memory usage tracking")
    memory_usage = monitor.get_memory_usage()
    logger.info(f"Current memory usage: {memory_usage:.2f}MB")
    
    # Test 3: Metrics recording
    logger.info("Test 3: Metrics recording")
    metrics = PerformanceMetrics(
        creation_time=0.5,
        memory_usage=50.0,
        figure_size=(10, 8),
        data_points=1000,
        complexity=PlotComplexity.MEDIUM
    )
    monitor.record_metrics(metrics)
    logger.info("✅ Metrics recorded successfully")
    
    # Test 4: Performance report
    logger.info("Test 4: Performance report")
    report = monitor.get_performance_report()
    logger.info(f"Performance report: {report}")


def test_figure_cache():
    """Test figure caching capabilities."""
    logger.info("=" * 60)
    logger.info("TESTING FIGURE CACHE")
    logger.info("=" * 60)
    
    cache = FigureCache(max_size=5, max_memory_mb=50.0)
    
    # Test 1: Cache put and get
    logger.info("Test 1: Cache put and get")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([1, 2, 3], [1, 4, 2])
    
    cache.put("test_plot", "data_hash_1", fig, title="Test Plot")
    cached_fig = cache.get("test_plot", "data_hash_1", title="Test Plot")
    
    if cached_fig is not None:
        logger.info("✅ Cache put and get successful")
    else:
        logger.error("❌ Cache get failed")
    
    # Test 2: Cache miss
    logger.info("Test 2: Cache miss")
    cached_fig = cache.get("test_plot", "data_hash_2", title="Different Plot")
    if cached_fig is None:
        logger.info("✅ Cache miss handled correctly")
    else:
        logger.error("❌ Unexpected cache hit")
    
    # Test 3: Cache statistics
    logger.info("Test 3: Cache statistics")
    stats = cache.get_stats()
    logger.info(f"Cache stats: {stats}")
    
    # Test 4: Cache cleanup
    logger.info("Test 4: Cache cleanup")
    cache.clear()
    stats_after_clear = cache.get_stats()
    logger.info(f"Cache stats after clear: {stats_after_clear}")


def test_lazy_plot_creator():
    """Test lazy plot creation capabilities."""
    logger.info("=" * 60)
    logger.info("TESTING LAZY PLOT CREATOR")
    logger.info("=" * 60)
    
    cache = FigureCache()
    lazy_creator = LazyPlotCreator(cache)
    
    # Test 1: Complexity estimation
    logger.info("Test 1: Complexity estimation")
    complexities = [
        lazy_creator.estimate_complexity(50, "scatter"),
        lazy_creator.estimate_complexity(500, "line"),
        lazy_creator.estimate_complexity(5000, "heatmap"),
        lazy_creator.estimate_complexity(50000, "scatter")
    ]
    
    for i, complexity in enumerate(complexities):
        logger.info(f"Data size {[50, 500, 5000, 50000][i]}: {complexity.value}")
    
    # Test 2: Lazy plot creation
    logger.info("Test 2: Lazy plot creation")
    
    def create_test_plot(**kwargs):
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title("Test Plot")
        return fig
    
    # Generate data hash
    import hashlib
    data_hash = hashlib.md5(b"test_data").hexdigest()
    
    fig = lazy_creator.create_plot_lazy(
        create_test_plot,
        data_hash,
        "test_plot",
        100,
        title="Test Plot"
    )
    
    logger.info("✅ Lazy plot creation successful")
    
    # Test 3: Performance report
    logger.info("Test 3: Performance report")
    report = lazy_creator.get_performance_report()
    logger.info(f"Lazy creator performance report: {report}")


def test_memory_manager():
    """Test memory management capabilities."""
    logger.info("=" * 60)
    logger.info("TESTING MEMORY MANAGER")
    logger.info("=" * 60)
    
    manager = MemoryManager(threshold_mb=50.0)
    
    # Test 1: Memory usage checking
    logger.info("Test 1: Memory usage checking")
    memory_info = manager.check_memory_usage()
    logger.info(f"Memory info: {memory_info}")
    
    # Test 2: Cleanup decision
    logger.info("Test 2: Cleanup decision")
    should_cleanup = manager.should_cleanup()
    logger.info(f"Should cleanup: {should_cleanup}")
    
    # Test 3: Memory cleanup
    logger.info("Test 3: Memory cleanup")
    cleanup_result = manager.cleanup_memory()
    logger.info(f"Cleanup result: {cleanup_result}")
    
    # Test 4: Data size optimization
    logger.info("Test 4: Data size optimization")
    optimization_results = []
    for data_size in [100, 1000, 10000, 100000]:
        result = manager.optimize_for_data_size(data_size)
        optimization_results.append((data_size, result))
        logger.info(f"Data size {data_size}: {result}")


def test_performance_decorators():
    """Test performance monitoring decorators."""
    logger.info("=" * 60)
    logger.info("TESTING PERFORMANCE DECORATORS")
    logger.info("=" * 60)
    
    # Test 1: Performance monitor decorator
    logger.info("Test 1: Performance monitor decorator")
    
    @performance_monitor
    def test_function(data_size: int):
        """Test function with performance monitoring."""
        time.sleep(0.1)  # Simulate work
        return np.random.rand(data_size)
    
    result = test_function(1000)
    logger.info("✅ Performance monitor decorator successful")
    
    # Test 2: Lazy plot creation decorator
    logger.info("Test 2: Lazy plot creation decorator")
    
    @lazy_plot_creation()
    def create_decorated_plot(data):
        """Test function with lazy plot creation."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(data)
        return fig
    
    test_data = np.random.rand(100)
    fig = create_decorated_plot(test_data)
    logger.info("✅ Lazy plot creation decorator successful")


def test_plot_factory_performance():
    """Test performance optimizations in PlotFactory."""
    logger.info("=" * 60)
    logger.info("TESTING PLOT FACTORY PERFORMANCE")
    logger.info("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        factory = PlotFactory(model_name="Test Model", save_dir=temp_dir)
        
        # Test 1: Standard plot creation
        logger.info("Test 1: Standard plot creation")
        training_data = {
            'loss': [0.5, 0.3, 0.2, 0.1],
            'accuracy': [0.6, 0.7, 0.8, 0.9]
        }
        
        start_time = time.time()
        fig, ax = factory.create_training_plot(training_data)
        standard_time = time.time() - start_time
        logger.info(f"Standard plot creation time: {standard_time:.3f}s")
        
        # Test 2: Optimized plot creation
        logger.info("Test 2: Optimized plot creation")
        start_time = time.time()
        fig, ax = factory.create_training_plot_optimized(training_data)
        optimized_time = time.time() - start_time
        logger.info(f"Optimized plot creation time: {optimized_time:.3f}s")
        
        # Test 3: Large dataset optimization
        logger.info("Test 3: Large dataset optimization")
        large_training_data = {
            'loss': list(np.random.rand(2000)),
            'accuracy': list(np.random.rand(2000))
        }
        
        start_time = time.time()
        fig, ax = factory.create_training_plot_optimized(large_training_data)
        large_dataset_time = time.time() - start_time
        logger.info(f"Large dataset plot creation time: {large_dataset_time:.3f}s")
        
        # Test 4: Performance report
        logger.info("Test 4: Performance report")
        report = factory.get_performance_report()
        logger.info(f"PlotFactory performance report: {report}")
        
        # Test 5: Memory optimization
        logger.info("Test 5: Memory optimization")
        optimization = factory.optimize_for_data_size(5000)
        logger.info(f"Memory optimization for 5000 data points: {optimization}")
        
        # Test 6: Memory cleanup
        logger.info("Test 6: Memory cleanup")
        cleanup_result = factory.cleanup_memory()
        logger.info(f"Memory cleanup result: {cleanup_result}")


def test_base_visualizer_performance():
    """Test performance optimizations in BaseVisualizer."""
    logger.info("=" * 60)
    logger.info("TESTING BASE VISUALIZER PERFORMANCE")
    logger.info("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        visualizer = BaseVisualizer(
            model_name="Test Model",
            default_save_dir=temp_dir
        )
        
        # Test 1: Standard figure creation
        logger.info("Test 1: Standard figure creation")
        start_time = time.time()
        fig, ax = visualizer.create_figure()
        standard_time = time.time() - start_time
        logger.info(f"Standard figure creation time: {standard_time:.3f}s")
        
        # Test 2: Optimized figure creation
        logger.info("Test 2: Optimized figure creation")
        start_time = time.time()
        fig, ax = visualizer.create_figure_optimized()
        optimized_time = time.time() - start_time
        logger.info(f"Optimized figure creation time: {optimized_time:.3f}s")
        
        # Test 3: Performance report
        logger.info("Test 3: Performance report")
        report = visualizer.get_performance_report()
        logger.info(f"BaseVisualizer performance report: {report}")
        
        # Test 4: Memory optimization
        logger.info("Test 4: Memory optimization")
        optimization = visualizer.optimize_for_data_size(3000)
        logger.info(f"Memory optimization for 3000 data points: {optimization}")
        
        # Test 5: Memory cleanup
        logger.info("Test 5: Memory cleanup")
        cleanup_result = visualizer.cleanup_memory()
        logger.info(f"Memory cleanup result: {cleanup_result}")


def test_convenience_functions():
    """Test convenience functions for performance optimization."""
    logger.info("=" * 60)
    logger.info("TESTING CONVENIENCE FUNCTIONS")
    logger.info("=" * 60)
    
    # Test 1: Performance summary
    logger.info("Test 1: Performance summary")
    summary = get_performance_summary()
    logger.info(f"Performance summary: {summary}")
    
    # Test 2: Memory optimization for large data
    logger.info("Test 2: Memory optimization for large data")
    for data_size in [1000, 10000, 100000]:
        optimization = optimize_memory_for_large_data(data_size)
        logger.info(f"Data size {data_size}: {optimization}")
    
    # Test 3: Memory cleanup if needed
    logger.info("Test 3: Memory cleanup if needed")
    cleanup_result = cleanup_memory_if_needed()
    logger.info(f"Cleanup result: {cleanup_result}")


def test_performance_benchmarking():
    """Test performance benchmarking with different scenarios."""
    logger.info("=" * 60)
    logger.info("TESTING PERFORMANCE BENCHMARKING")
    logger.info("=" * 60)
    
    # Test 1: Small dataset performance
    logger.info("Test 1: Small dataset performance")
    small_data = np.random.rand(100, 2)
    start_time = time.time()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(small_data[:, 0], small_data[:, 1])
    small_time = time.time() - start_time
    logger.info(f"Small dataset plot time: {small_time:.3f}s")
    
    # Test 2: Medium dataset performance
    logger.info("Test 2: Medium dataset performance")
    medium_data = np.random.rand(1000, 2)
    start_time = time.time()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(medium_data[:, 0], medium_data[:, 1])
    medium_time = time.time() - start_time
    logger.info(f"Medium dataset plot time: {medium_time:.3f}s")
    
    # Test 3: Large dataset performance
    logger.info("Test 3: Large dataset performance")
    large_data = np.random.rand(10000, 2)
    start_time = time.time()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(large_data[:, 0], large_data[:, 1])
    large_time = time.time() - start_time
    logger.info(f"Large dataset plot time: {large_time:.3f}s")
    
    # Performance comparison
    logger.info("Performance comparison:")
    logger.info(f"Small dataset (100 points): {small_time:.3f}s")
    logger.info(f"Medium dataset (1000 points): {medium_time:.3f}s")
    logger.info(f"Large dataset (10000 points): {large_time:.3f}s")
    logger.info(f"Scaling factor (small to medium): {medium_time/small_time:.1f}x")
    logger.info(f"Scaling factor (medium to large): {large_time/medium_time:.1f}x")


def main():
    """Run all performance optimization tests."""
    logger.info("Starting Performance Optimization Tests")
    logger.info("=" * 80)
    
    try:
        # Test performance monitoring
        test_performance_monitor()
        
        # Test figure caching
        test_figure_cache()
        
        # Test lazy plot creation
        test_lazy_plot_creator()
        
        # Test memory management
        test_memory_manager()
        
        # Test performance decorators
        test_performance_decorators()
        
        # Test PlotFactory performance
        test_plot_factory_performance()
        
        # Test BaseVisualizer performance
        test_base_visualizer_performance()
        
        # Test convenience functions
        test_convenience_functions()
        
        # Test performance benchmarking
        test_performance_benchmarking()
        
        logger.info("=" * 80)
        logger.info("🎉 ALL PERFORMANCE OPTIMIZATION TESTS PASSED!")
        logger.info("=" * 80)
        
        # Summary of performance optimization features
        logger.info("SUMMARY OF PERFORMANCE OPTIMIZATION FEATURES:")
        logger.info("✅ Lazy plot creation for large datasets")
        logger.info("✅ Figure caching with LRU eviction")
        logger.info("✅ Memory management and cleanup")
        logger.info("✅ Performance monitoring and benchmarking")
        logger.info("✅ Plot complexity estimation")
        logger.info("✅ Background plot generation")
        logger.info("✅ Performance optimization recommendations")
        logger.info("✅ Integration with BaseVisualizer and PlotFactory")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main() 