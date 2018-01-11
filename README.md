# MLADV_finalpj
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

1. ppca_weng 包含两个版本的EM PPCA，simple_PPCA类和PPCA。

  + simple_PPCA: 不考虑sigma，在EM迭代更新中，运算皆能使用矩阵运算进行加速。
  
  + PPCA: 考虑了sigma的EM PPCA，由于在更新sigma的时候，其中有一项需要根据对应不同点计算迹，且其中包含矩阵乘法无法进行分解提取，因此需要单独计算Z[z_nz_N^T]。
