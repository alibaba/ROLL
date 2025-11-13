import React, { useEffect } from 'react';
import { Button } from 'antd';
import clsx from 'clsx';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

import styles from './index.module.css';
import HomeContent from '../components/HomeContent';

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className={styles.buttons}>
          <img src="img/roll.jpeg" alt="ROLL" style={{ maxWidth: '50%' }} />
        </div>
        <div className={styles.content}>
          {/* <h1 className="hero__title">{siteConfig.title}</h1> */}
          {/* <p className="hero__subtitle">{siteConfig.tagline}</p> */}
          <div className={styles.left}>
            <div className={styles.desc}>
              ROLL is an efficient and user-friendly RL library designed for Large Language Models (LLMs) utilizing Large Scale GPU resources. It significantly enhances LLM performance in key areas such as human preference alignment, complex reasoning, and multi-turn agentic interaction scenarios.
            </div>
            <div>
              <Button style={{ height: '60px', fontSize: '20px', padding: '0 20px' }} size="large" type="primary" href="/ROLL/zh-Hans/docs/QuickStart/multi_nodes_quick_start_cn">Get started</Button>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <main>
        <HomeContent />
      </main>
    </Layout>
  );
}
