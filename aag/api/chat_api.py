//用户的问题输入
{
  "mode":"expert/normal" ,
  "model":"qwen3-max,
  "dataset":"xxxx",
  "message":"用户输入的具体内容...
}

//用户提出修改dag
{
  "modify": True / False,
  "suggestions": "我认为缺少了梅赛德斯奔驰..." 
}

//后端返回dag结构
data: {
  "type": "result",
  "contentType": "dag",
  "content": {
 	 "nodes": [
    		{"id": "1", "label": "用户问题：分析文档", "tasktype":"Graph Algorithm (Pagerank)"},
   		{"id": "2", "label": "提取关键词", "tasktype":"Graph Algorithm (Pagerank)"},
    		{"id": "3", "label": "检索知识库", "tasktype":"Graph Algorithm (Pagerank)"}
         ],
  	"edges": [
		{"from": "1", "to": "2"},
        	{"from": "2", "to": "3"},
        	{"from": "3", "to": "4"}
  	]
    }
}

//后端返回文本
data: {
  "type": "thinking / result",
  "contentType": "text",
  "content": "用户问的是如何规划新年目标，需要分个人、家庭、职业三类..."
}

//后端返回代码
data: {
   "type": "thinking / result",
   "contentType": "code",
   "content": {
     "language": "python", 
     "code": "def analyze_goals(goals):\n    return [g for g in goals if g['priority'] > 3]" 
   }
 }