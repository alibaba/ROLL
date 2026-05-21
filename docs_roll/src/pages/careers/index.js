import React from 'react';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

import styles from './index.module.css';

const publications = [
  {
    title: 'NebulaSQL: A Large-scale Feature Computation System for Online Recommendation',
    note: 'SIGMOD 2026 Industry',
    venue: "SIGMOD'26",
    href: 'https://2026.sigmod.org/sigmod_industry_papers.shtml',
  },
  {
    title: 'Weave: Efficient Co-Scheduling for Disaggregated RL Post-Training',
    note: 'OSDI 2026',
    venue: "OSDI'26",
    href: 'https://www.usenix.org/conference/osdi26/presentation/wu-tianyuan',
  },
  {
    title: 'RollArt: Scaling Agentic RL Training via Disaggregated Infrastructure',
    note: 'OSDI 2026',
    venue: "OSDI'26",
    href: 'https://arxiv.org/abs/2512.22560',
  },
  {
    title: 'RollPacker: Mitigating Long-Tail Rollouts for Fast, Synchronous RL Post-Training',
    note: 'NSDI 2026 Fall',
    venue: "NSDI'26",
    href: 'https://arxiv.org/abs/2509.21009',
  },
  {
    title: 'Attack of the Bubbles: Straggler-Resilient Pipeline Parallelism for Large Model Training',
    note: 'NSDI 2026 Spring · 🏆 Outstanding Paper Award',
    venue: "NSDI'26",
    highlight: true,
    href: 'https://www.usenix.org/conference/nsdi26/presentation/wu-tianyuan',
  },
  {
    title: 'GREYHOUND: Hunting Fail-Slows in Hybrid-Parallel Training at Scale',
    note: 'USENIX ATC 2025',
    venue: "ATC'25",
    href: 'https://www.usenix.org/conference/atc25/presentation/wu-tianyuan',
  },
  {
    title: 'FaPES: Enabling Efficient Elastic Scaling for Serverless Machine Learning Platforms',
    note: 'SoCC 2024',
    venue: "SoCC'24",
    href: 'https://doi.org/10.1145/3698038.3698548',
  },
  {
    title: 'GBA: A General, Flexible, and Scalable Batch Auction System for Data Centers',
    note: 'NeurIPS 2022',
    venue: "NeurIPS'22",
    href: 'https://arxiv.org/abs/2205.11048',
  },
  {
    title: 'PICASSO: Unleashing the Potential of GPU-centric Training for Wide-and-deep Recommender Systems',
    note: 'ICDE 2022',
    venue: "ICDE'22",
    href: 'https://arxiv.org/abs/2204.04903',
  },
];

const projects = [
  {
    name: 'Megatron-LLaMA',
    tags: ['LLM', 'PyTorch'],
    href: 'https://github.com/alibaba/Megatron-LLaMA',
    zhDesc: '基于 Megatron 的大语言模型开源训练框架，支持高效分布式 LLM 训练。',
    enDesc: 'An open-source LLM training framework based on Megatron, supporting efficient distributed training.',
  },
  {
    name: 'X-DeepLearning (XDL)',
    tags: ['Sparse', 'RecSys'],
    href: 'https://github.com/alibaba/x-deeplearning',
    zhDesc: '阿里巴巴开源的稀疏模型训练框架，支持大规模推荐和广告场景。',
    enDesc: "Alibaba's open-source sparse model training framework for large-scale recommendation and advertising.",
  },
  {
    name: 'ROLL',
    tags: ['RL', 'Distributed'],
    href: 'https://github.com/alibaba/ROLL',
    zhDesc: '强化学习开源训练框架，支持大规模 RL Post-Training 的高效分布式执行。',
    enDesc: 'An open-source RL training framework for efficient distributed RL post-training at scale.',
  },
  {
    name: 'Euler',
    tags: ['GNN', 'Graph'],
    href: 'https://github.com/alibaba/euler',
    zhDesc: '阿里巴巴开源的分布式图学习引擎，支持大规模图神经网络训练。',
    enDesc: "Alibaba's open-source distributed graph learning engine for large-scale GNN training.",
  },
  {
    name: 'RecIS',
    tags: ['RecSys', 'Training'],
    href: 'https://github.com/alibaba/RecIS',
    zhDesc: '预估大模型训练框架，面向推荐和广告场景的工业级大模型训练系统。',
    enDesc: 'A large model training framework for recommendation and advertising at industrial scale.',
  },
];

const copy = {
  zh: {
    title: '阿里巴巴智能引擎算法平台团队',
    description:
      '阿里控股集团智能引擎事业部算法平台团队负责构建阿里集团模型训练基础设施，承担HappyHorse、HappyOyster系列模型的数据和训练Infra建设。',
    heroTitle: '阿里巴巴智能引擎算法平台团队',
    heroDesc:
      '阿里控股集团智能引擎事业部算法平台团队负责构建阿里集团模型训练基础设施，承担HappyHorse、HappyOyster系列模型的数据和训练Infra建设。团队建设了业界一流的大语言模型、多模态模型、生成模型的预训练、后训练框架以及样本存储和计算系统。开源项目包括Megatron-LLaMA、ROLL、RecIS 等，在NSDI、OSDI、SIGMOD等顶级会议发布了多篇工作，并获得 26 年 NSDI Outstanding Paper Award。团队致力于通过分布式优化、软硬件结合、模型-Infra Codesign等手段，从数据处理到训练全面优化大模型迭代效率，提升模型效果上限，打造行业前沿大模型基础设施。',
    aboutTag: '关于团队',
    aboutTitle: '使命与职责',
    aboutDesc:
      '我们负责阿里巴巴大规模训练基础设施的构建，涵盖大语言模型训练 Infra、多模态大模型训练 Infra、预估算法大模型训练 Infra、特征计算与处理 Infra 以及算法平台建设等关键领域。',
    publicationsTag: '学术成果',
    publicationsTitle: '代表性论文',
    openSourceTag: '开源项目',
    openSourceTitle: '开源训练框架与系统',
  },
  en: {
    title: 'Alibaba Intelligent Engine Algorithm Platform Team',
    description:
      "Building Alibaba Group's model training infrastructure for the HappyHorse and HappyOyster model families.",
    heroTitle: 'Alibaba Intelligent Engine Algorithm Platform Team',
    heroDesc:
      "The Algorithm Platform team of Alibaba Holding Group's Intelligent Engine Business Unit builds Alibaba Group's model training infrastructure and is responsible for data and training Infra for the HappyHorse and HappyOyster model families. The team maintains industry-leading pre-training and post-training frameworks for large language models, multimodal models, and generative models, together with sample storage and compute systems. Open source projects include Megatron-LLaMA, ROLL, and RecIS. The team has published multiple works at top conferences including NSDI, OSDI, and SIGMOD, and received the 2026 NSDI Outstanding Paper Award. Through distributed optimization, software-hardware co-design, and model-Infra codesign, the team optimizes large model iteration efficiency from data processing through training, expands the ceiling of model quality, and builds frontier infrastructure for large models.",
    aboutTag: 'About Us',
    aboutTitle: 'Mission & Responsibilities',
    aboutDesc:
      "We are responsible for building Alibaba's large-scale training infrastructure, covering LLM training Infra, multi-modal large model training Infra, recommendation algorithm model training Infra, feature computation & processing Infra, and algorithm platform construction.",
    publicationsTag: 'Publications',
    publicationsTitle: 'Selected Publications',
    openSourceTag: 'Open Source',
    openSourceTitle: 'Open Source Frameworks',
  },
};

export default function Careers() {
  const { i18n } = useDocusaurusContext();
  const isChinese = i18n.currentLocale !== 'en';
  const text = isChinese ? copy.zh : copy.en;

  return (
    <Layout title={text.title} description={text.description}>
      <main className={styles.page}>
        <section className={styles.hero}>
          <h1>{text.heroTitle}</h1>
          <p>{text.heroDesc}</p>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionTag}>{text.aboutTag}</div>
          <h2>{text.aboutTitle}</h2>
          <p className={styles.sectionDesc}>{text.aboutDesc}</p>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionTag}>{text.publicationsTag}</div>
          <h2>{text.publicationsTitle}</h2>
          <div className={styles.publicationList}>
            {publications.map((publication) => {
              const title = publication.href ? (
                <a href={publication.href} target="_blank" rel="noreferrer">
                  {publication.title}
                </a>
              ) : (
                publication.title
              );
              const note =
                publication.highlight && publication.note.includes('🏆') ? (
                  <>
                    {publication.note.split('🏆')[0]}
                    <span className={styles.awardEmoji} role="img" aria-label="Outstanding Paper Award">
                      🏆
                    </span>
                    {publication.note.split('🏆')[1]}
                  </>
                ) : (
                  publication.note
                );

              return (
                <article className={styles.publicationRow} key={publication.title}>
                  <div className={styles.publicationMain}>
                    <h3>{title}</h3>
                    <p className={publication.highlight ? styles.highlightNote : undefined}>{note}</p>
                  </div>
                  <div className={styles.venue}>{publication.venue}</div>
                </article>
              );
            })}
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionTag}>{text.openSourceTag}</div>
          <h2>{text.openSourceTitle}</h2>
          <div className={styles.projectGrid}>
            {projects.map((project) => (
              <article className={styles.projectCard} key={project.name}>
                <h3>{project.name}</h3>
                <p>{isChinese ? project.zhDesc : project.enDesc}</p>
                <div className={styles.projectFooter}>
                  <div className={styles.projectTags}>
                    {project.tags.map((tag) => (
                      <span className={styles.projectTag} key={tag}>
                        {tag}
                      </span>
                    ))}
                  </div>
                  <a href={project.href} target="_blank" rel="noreferrer">
                    GitHub
                  </a>
                </div>
              </article>
            ))}
          </div>
        </section>
      </main>
    </Layout>
  );
}
