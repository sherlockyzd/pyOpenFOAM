#!/usr/bin/env python3
"""
METIS网格分区器 - 一体化解决方案
集成网格分区、区域分解和分布式求解功能
"""

import numpy as np
from mpi4py import MPI
from abc import ABC, abstractmethod

# 尝试导入METIS库
try:
    import metis
    METIS_AVAILABLE = True
    METIS_BACKEND = 'metis'
except ImportError:
    try:
        import pymetis
        METIS_AVAILABLE = True
        METIS_BACKEND = 'pymetis'
    except ImportError:
        METIS_AVAILABLE = False
        METIS_BACKEND = None
        print("警告: METIS库不可用，请安装: pip install metis 或 pip install pymetis")

class MeshPartitioner(ABC):
    """网格分区器抽象基类"""
    
    @abstractmethod
    def partition(self, mesh, num_partitions):
        """对网格进行分区"""
        pass

class GeometricPartitioner(MeshPartitioner):
    """简单几何分区器 - 作为METIS的备选"""
    
    def partition(self, mesh, num_partitions):
        """基于坐标的简单分区"""
        n_elements = mesh.numberOfElements
        
        if num_partitions == 1:
            return np.zeros(n_elements, dtype=np.int32)
        
        if hasattr(mesh, 'elementCentroids'):
            centroids = mesh.elementCentroids.value
            # 找到变化最大的方向
            ranges = np.max(centroids, axis=0) - np.min(centroids, axis=0)
            max_dir = np.argmax(ranges)
            
            # 沿最大方向排序分区
            sorted_indices = np.argsort(centroids[:, max_dir])
        else:
            # 没有坐标信息，简单按索引分区
            sorted_indices = np.arange(n_elements)
        
        # 均匀分配
        partition_map = np.zeros(n_elements, dtype=np.int32)
        elements_per_partition = n_elements // num_partitions
        remainder = n_elements % num_partitions
        
        start_idx = 0
        for p in range(num_partitions):
            current_size = elements_per_partition + (1 if p < remainder else 0)
            end_idx = start_idx + current_size
            partition_map[sorted_indices[start_idx:end_idx]] = p
            start_idx = end_idx
        
        print(f"几何分区完成: {num_partitions}个分区")
        return partition_map

class METISPartitioner(MeshPartitioner):
    """基于METIS的高质量网格分区器"""
    
    def __init__(self, **metis_options):
        """
        初始化METIS分区器
        
        Args:
            **metis_options: METIS参数选项
        """
        if not METIS_AVAILABLE:
            raise ImportError("METIS库不可用，请安装: pip install metis")
        
        # 高质量的默认参数
        self.metis_options = {
            'seed': 42,           # 确保结果可重复
            'niter': 10,          # 足够的迭代次数
            'ufactor': 30,        # 允许3%的不平衡
            **metis_options
        }
        
        if METIS_BACKEND:
            print(f"METIS分区器初始化完成，后端: {METIS_BACKEND}")
        
    def partition(self, mesh, num_partitions):
        """
        使用METIS对网格进行分区
        
        Args:
            mesh: Polymesh对象
            num_partitions: 分区数量
            
        Returns:
            partition_map: 每个单元的分区ID数组
        """
        if num_partitions == 1:
            return np.zeros(mesh.numberOfElements, dtype=np.int32)
        
        # 构建METIS格式的图
        adjacency_list, weights = self._build_metis_graph(mesh)
        
        # 执行分区
        try:
            if METIS_BACKEND == 'metis':
                partition_map = self._partition_with_metis(adjacency_list, weights, num_partitions)
            else:  # pymetis
                partition_map = self._partition_with_pymetis(adjacency_list, weights, num_partitions)
            
            # 分析分区质量
            self._analyze_partition_quality(mesh, partition_map, num_partitions)
            return partition_map
            
        except Exception as e:
            print(f"METIS分区失败: {e}，使用几何分区")
            fallback = GeometricPartitioner()
            return fallback.partition(mesh, num_partitions)
    
    def _build_metis_graph(self, mesh):
        """
        构建METIS需要的图结构
        
        Args:
            mesh: Polymesh对象
            
        Returns:
            adjacency_list: 邻接表
            weights: 单元权重（可选）
        """
        n_elements = mesh.numberOfElements
        
        # 构建邻接表
        adjacency_list = []
        for i in range(n_elements):
            neighbors = mesh.elementNeighbours[i]
            # 过滤掉无效的邻居
            valid_neighbors = [nb for nb in neighbors if 0 <= nb < n_elements and nb != i]
            adjacency_list.append(valid_neighbors)
        
        # 单元权重（基于体积）
        if hasattr(mesh, 'elementVolumes'):
            # 将体积归一化为整数权重
            volumes = mesh.elementVolumes.value
            max_vol = np.max(volumes)
            min_vol = np.min(volumes)
            
            if max_vol > min_vol:
                # 归一化到1-100的整数范围
                normalized = (volumes - min_vol) / (max_vol - min_vol) * 99 + 1
                weights = normalized.astype(np.int32)
            else:
                weights = np.ones(n_elements, dtype=np.int32)
        else:
            weights = np.ones(n_elements, dtype=np.int32)
        
        return adjacency_list, weights
    
    def _partition_with_metis(self, adjacency_list, weights, num_partitions):
        """使用metis库进行分区"""
        # 构建NetworkX图
        G = self._adjacency_to_networkx(adjacency_list)
        
        # 选择最佳分区策略
        if num_partitions == 2:
            # 二分法最优
            cuts, parts = metis.part_graph(G, nparts=2, recursive=True, **self.metis_options)
        elif num_partitions <= 8:
            # k-way方法适合中等分区数
            cuts, parts = metis.part_graph(G, nparts=num_partitions, recursive=False, **self.metis_options)
        else:
            # 大分区数用递归
            cuts, parts = metis.part_graph(G, nparts=num_partitions, recursive=True, **self.metis_options)
        
        partition_map = np.array(parts, dtype=np.int32)
        print(f"METIS分区完成，切边数: {cuts}")
        return partition_map
    
    def _partition_with_pymetis(self, adjacency_list, weights, num_partitions):
        """使用pymetis库进行分区"""
        # 转换为CSR格式
        xadj = [0]
        adjncy = []
        
        for neighbors in adjacency_list:
            adjncy.extend(neighbors)
            xadj.append(len(adjncy))
        
        # 执行分区
        cuts, parts = pymetis.part_graph(
            nparts=num_partitions,
            adjacency=(xadj, adjncy),
            node_weights=weights,
            recursive=(num_partitions <= 8)  # 小分区数用递归
        )
        
        partition_map = np.array(parts, dtype=np.int32)
        print(f"pyMETIS分区完成，切边数: {cuts}")
        return partition_map
    
    def _adjacency_to_networkx(self, adjacency_list):
        """转换为NetworkX图"""
        try:
            import networkx as nx
            
            G = nx.Graph()
            G.add_nodes_from(range(len(adjacency_list)))
            
            # 添加边（避免重复）
            for i, neighbors in enumerate(adjacency_list):
                for neighbor in neighbors:
                    if neighbor > i:
                        G.add_edge(i, neighbor)
            
            return G
        except ImportError:
            raise ImportError("使用metis库需要安装NetworkX: pip install networkx")
    
    def _analyze_partition_quality(self, mesh, partition_map, num_partitions):
        """分析分区质量"""
        n_elements = mesh.numberOfElements
        
        # 分区大小统计
        partition_sizes = [np.sum(partition_map == p) for p in range(num_partitions)]
        avg_size = n_elements / num_partitions
        max_imbalance = max(partition_sizes) / avg_size
        min_fill = min(partition_sizes) / avg_size
        
        # 计算切边数
        cut_edges = 0
        for i in range(n_elements):
            my_partition = partition_map[i]
            neighbors = mesh.elementNeighbours[i]
            
            for neighbor in neighbors:
                if 0 <= neighbor < n_elements and partition_map[neighbor] != my_partition:
                    cut_edges += 1
        
        cut_edges //= 2  # 避免重复计算
        
        print(f"\n=== METIS分区质量报告 ===")
        print(f"总单元数: {n_elements:,}")
        print(f"分区数: {num_partitions}")
        print(f"各分区大小: {partition_sizes}")
        print(f"负载不均衡度: {max_imbalance:.2f}")
        print(f"切边数: {cut_edges:,} ({cut_edges/n_elements:.1%})")
        
        # 质量评估
        if max_imbalance > 1.3:
            print("⚠️  负载不均衡较严重")
        if cut_edges/n_elements > 0.05:
            print("⚠️  切边比例偏高")
        else:
            print("✅ 分区质量良好")

class AdaptiveMETISPartitioner(METISPartitioner):
    """自适应METIS分区器 - 根据网格特征智能选择参数"""
    
    def partition(self, mesh, num_partitions):
        n_elements = mesh.numberOfElements
        
        # 根据网格大小自动选择策略
        if n_elements < 1000:
            # 小网格: 使用简单几何分区
            print(f"小网格({n_elements}单元), 使用几何分区")
            fallback = GeometricPartitioner()
            return fallback.partition(mesh, num_partitions)
        
        # 中大网格: 使用METIS
        if n_elements < 10000:
            self.metis_options.update({'niter': 5, 'ufactor': 50})
            print(f"中等网格({n_elements}单元), 使用快速METIS")
        else:
            self.metis_options.update({'niter': 10, 'ufactor': 30})
            print(f"大网格({n_elements}单元), 使用高质量METIS")
        
        return super().partition(mesh, num_partitions)

def create_partitioner(method='auto', **options):
    """
    创建METIS分区器的简化工厂函数
    
    Args:
        method: 'auto'(推荐), 'metis', 'adaptive_metis', 'geometric'
        **options: METIS参数
    """
    if method == 'auto' or method == 'adaptive_metis':
        if METIS_AVAILABLE:
            return AdaptiveMETISPartitioner(**options)
        else:
            print("METIS不可用，使用几何分区")
            return GeometricPartitioner()
    
    elif method == 'metis':
        return METISPartitioner(**options)
    
    elif method == 'geometric':
        return GeometricPartitioner()
    
    else:
        raise ValueError(f"不支持的分区方法: {method}")

# 使用示例和测试
if __name__ == "__main__":
    print("METIS网格分区器 - 简化版")
    print(f"METIS可用: {METIS_AVAILABLE}")
    print(f"后端: {METIS_BACKEND}")
    
    if METIS_AVAILABLE:
        partitioner = create_partitioner('auto')
        print(f"分区器创建成功: {type(partitioner).__name__}")
        print("使用方法:")
        print("  partitioner = create_partitioner('auto')")
        print("  partition_map = partitioner.partition(mesh, 4)")
    else:
        print("请安装METIS: pip install metis")