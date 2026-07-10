import type { Node, Relation } from '@/api/initialization'

// 领域图谱提取模板：作为 LLM 实体关系提取的 few-shot 示例整体写入
// extract_config（text/tags/nodes/relations 四字段需同时非空，后端
// validateExtractConfig 强校验）。relations 的端点必须来自 nodes 的
// name、type 必须来自 tags，否则 UI 下拉回显不出来。
export interface GraphExtractTemplate {
  text: string
  tags: string[]
  nodes: Node[]
  relations: Relation[]
}

// 博物馆展品领域默认模板（本部署的主要业务领域）
export const MUSEUM_GRAPH_TEMPLATE: GraphExtractTemplate = {
  text:
    '《千里江山图》是北宋画家王希孟创作的绢本设色画，中国十大传世名画之一，' +
    '现收藏于北京故宫博物院。该画完成于宋徽宗政和三年（1113年），王希孟时年' +
    '十八岁。画卷以石青、石绿等矿物颜料绘成，属青绿山水一派，描绘了连绵起伏' +
    '的群山冈峦和烟波浩渺的江河湖水。',
  tags: ['创作者', '年代', '朝代', '材质', '收藏于', '出土于', '展出于', '类属'],
  nodes: [
    { name: '千里江山图', attributes: ['绢本设色画', '中国十大传世名画之一', '完成于1113年'] },
    { name: '王希孟', attributes: ['北宋画家', '创作此画时年十八岁'] },
    { name: '北宋', attributes: ['朝代'] },
    { name: '绢本设色', attributes: ['绘画材质与技法'] },
    { name: '青绿山水', attributes: ['以石青、石绿等矿物颜料为主的山水画流派'] },
    { name: '北京故宫博物院', attributes: ['博物馆', '位于北京'] },
  ],
  relations: [
    { node1: '千里江山图', node2: '王希孟', type: '创作者' },
    { node1: '千里江山图', node2: '北宋', type: '朝代' },
    { node1: '千里江山图', node2: '绢本设色', type: '材质' },
    { node1: '千里江山图', node2: '青绿山水', type: '类属' },
    { node1: '千里江山图', node2: '北京故宫博物院', type: '收藏于' },
  ],
}
