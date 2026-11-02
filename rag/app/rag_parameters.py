# 分块大小（字符）：越大上下文越完整，但检索可能不够精确；修改后需重新入库。
CHUNK_SIZE = 500
# 分块重叠（字符）：越大越能保留边界信息，但会增加重复；修改后需重新入库。
CHUNK_OVERLAP = 50

# 问题重写最多生成几条：通常用 1 条整理问题，多种表达交给多查询。
REWRITE_QUERY_LIMIT = 1
# 每轮最多检索几条不同查询：越多覆盖可能越广，但耗时增加。
QUERIES_PER_ROUND = 6
# 语义相近的向量检索取前几条：越多越可能找全，但处理开销可能增加。
DENSE_TOP_K = 20
# 关键词匹配的 BM25 检索取前几条：越多越可能找全，但处理开销可能增加。
BM25_TOP_K = 20
# 融合后每条查询保留多少结果：越大候选越多，但后续处理开销增加。
FUSED_TOP_K = 10
# RRF 融合常数：越小越重视靠前结果，越大不同排名的差距越平缓。
RRF_K = 60
# 最多将多少去重候选交给重排：越大覆盖可能越广，但重排更慢。
RERANK_CANDIDATES = 20
# 重排聚合后保留多少完整文档：越大上下文越多，但模型输入和耗时增加。
EVIDENCE_TOP_K = 5
# 证据不足时最多补检几轮：越多可能找全，但耗时和 token 增加。
SUPPLEMENTAL_ROUNDS = 1
# 单条检索失败后的重试次数：越多越能应对短暂故障，但失败时等待更久。
SEARCH_RETRIES = 1
# 模型输出不是 JSON 时的重试次数：越多越可能成功，但耗时和 token 增加。
JSON_RETRIES = 1
# RAG 模型及评测裁判请求失败后的 SDK 重试次数：越多失败时等待更久。
MODEL_REQUEST_RETRIES = 3
# 单次生成的最大输出 token：越大可回答得越长，但可能更慢且成本更高。
MAX_OUTPUT_TOKENS = 32768

# 参数只在 MCP 启动时读取；切片重叠必须小于切片长度。
if not (isinstance(CHUNK_SIZE, int) and isinstance(CHUNK_OVERLAP, int)
        and 0 <= CHUNK_OVERLAP < CHUNK_SIZE):
    raise ValueError("RAG 切片长度必须大于重叠，且重叠不能为负")
if any(not isinstance(value, int) or value <= 0 for value in (
        REWRITE_QUERY_LIMIT, QUERIES_PER_ROUND, DENSE_TOP_K, BM25_TOP_K, FUSED_TOP_K,
        RRF_K, RERANK_CANDIDATES, EVIDENCE_TOP_K, MAX_OUTPUT_TOKENS)):
    raise ValueError("RAG 查询数量、检索数量和 token 上限必须为正整数")
if EVIDENCE_TOP_K > RERANK_CANDIDATES:
    raise ValueError("RAG 最终证据数量不能大于重排候选数量")
if any(not isinstance(value, int) or value < 0 for value in (
        SUPPLEMENTAL_ROUNDS, SEARCH_RETRIES, JSON_RETRIES, MODEL_REQUEST_RETRIES)):
    raise ValueError("RAG 补检和重试次数必须为非负整数")
