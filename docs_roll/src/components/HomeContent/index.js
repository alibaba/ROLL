import React, { useState, useEffect, useRef } from 'react';
import { Button, Image, Divider, Col, Row, Collapse, Modal, ConfigProvider, theme } from 'antd';
import { GithubOutlined, WechatOutlined, XOutlined } from '@ant-design/icons';
import clsx from 'clsx';
import { useColorMode } from '@docusaurus/theme-common';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Translate from '@docusaurus/Translate';
import dayjs from 'dayjs';
import CountUp from 'react-countup';

import styles from './styles.module.css';

// Intersection Observer Hook
const useIntersectionObserver = (options = {}) => {
  const [isVisible, setIsVisible] = useState(false);
  const elementRef = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setIsVisible(true);
        observer.disconnect();
      }
    }, {
      threshold: 0.2,
      rootMargin: '0px',
      ...options
    });

    const currentElement = elementRef.current;
    if (currentElement) {
      observer.observe(currentElement);
    }

    return () => {
      if (currentElement) {
        observer.unobserve(currentElement);
      }
    };
  }, [options]);

  return [elementRef, isVisible];
};

export default () => {
  const [open, setOpen] = useState(false);
  const [todayStat, setTodayStat] = useState({});
  const today = dayjs().format('YYYY-MM-DD');
  const { colorMode } = useColorMode();
  const { i18n } = useDocusaurusContext();
  const { currentLocale } = i18n;
  const isChinese = currentLocale !== 'en';
  const targetPath = isChinese ? '/ROLL/zh-Hans/' : '/ROLL/'

  // Intersection Observer refs
  const [mainImgRef, mainImgVisible] = useIntersectionObserver();
  const [overviewRef, overviewVisible] = useIntersectionObserver();
  const [statsRef, statsVisible] = useIntersectionObserver();
  const [chooseRef, chooseVisible] = useIntersectionObserver();
  const [coreRef, coreVisible] = useIntersectionObserver();
  const [researchRef, researchVisible] = useIntersectionObserver();

  useEffect(() => {
    fetch('/ROLL/stats.json').then(res => res.json()).then(data => {
      setTodayStat(data[today]);
    })
  }, []);

  return <ConfigProvider theme={{ algorithm: colorMode === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm }}>
    <div className={clsx('container', styles.container)} id="home">
      <div>
        <div className={styles.subTitle}>
          <Translate>Open Source Framework · Powerful & Easy</Translate>
        </div>
        <div className={styles.title}>
          <div className={styles.names}>
            Reinforcement Learning
          </div>&nbsp;
          <div className={styles.names}>Optimization</div>&nbsp;
          for&nbsp;
          <div className={styles.names}>Large-scale</div>&nbsp;
          <div className={styles.names}>Learning</div>
        </div>
        {
          isChinese &&
          <div className={styles.title}>
            面向大规模学习的强化学习优化框架
          </div>
        }
        <div className={styles.desc}>
          <Translate>
            An open-source reinforcement learning library by Alibaba, optimized for large-scale language models. Supporting distributed training, multi-task learning, and agent interaction for simpler and more efficient AI model training.
          </Translate>
        </div>
        <div className={styles.buttons}>
          <Button href={`${targetPath}docs/Overview`} target='_blank' className={styles.btn}>
            <Translate>{"Get Started >"}</Translate>
          </Button>
          <Button className={styles.github} target='_blank' href="https://github.com/alibaba/ROLL" variant="outlined" icon={<GithubOutlined />}>{"Github >"}</Button>
          <Button className={styles.github} target='_blank' href="https://deepwiki.com/alibaba/ROLL" variant="outlined" icon={<Image width={14} src="https://img.alicdn.com/imgextra/i3/O1CN01JUBft41wYcExwXCOK_!!6000000006320-55-tps-460-500.svg" preview={false}></Image>}>{"DeepWiki >"}</Button>
        </div>

        <div ref={mainImgRef} className={clsx(styles.mainImg, styles.fadeIn, mainImgVisible && styles.visible)}>
          <Image className={styles.img} src="https://img.alicdn.com/imgextra/i2/O1CN01ZGW6zG1sXSmML15c3_!!6000000005776-2-tps-2160-1112.png" preview={false}></Image>
        </div>

        <div ref={overviewRef} className={clsx(styles.overview, styles.fadeIn, overviewVisible && styles.visible)}>
          <div className={styles.left}>
            <Image className={styles.img} src="https://img.alicdn.com/imgextra/i4/O1CN01t4tcSz1ZIN1sEzNnO_!!6000000003171-55-tps-54-54.svg" preview={false}></Image>
            <div>ROLL</div>
            <div>
              <Translate>Framework Overview</Translate>
            </div>
          </div>
          <div className={styles.right}>
            <Translate>
              ROLL (Reinforcement Learning Optimization for Large-scale Learning) is an open-source reinforcement learning framework by Alibaba, designed for large-scale language models. Built on Ray distributed architecture, supporting mainstream algorithms like PPO and GRPO, providing complete solutions from research to production.
            </Translate>
          </div>
        </div>

        <Divider style={{ borderColor: 'var(--home-divider-color)' }}></Divider>

        <div ref={statsRef} className={styles.stats}>
          <Row gutter={16}>
            <Col span={8}>
              <div className={styles.count}>
                {statsVisible && <CountUp end={todayStat?.stars || 2620} />}
              </div>
              <div className={styles.label}>
                <Translate>
                  Github Stars
                </Translate>
              </div>
            </Col>
            <Col span={8}>
              <div className={styles.count}>
                {statsVisible && <CountUp end={todayStat?.contributors || 37} />}
              </div>
              <div className={styles.label}>
                <Translate>
                  Contributors
                </Translate>
              </div>
            </Col>
            <Col span={8}>
              <div className={styles.count}>
                {statsVisible && <CountUp end={todayStat?.commits || 371} />}
              </div>
              <div className={styles.label}>
                <Translate>
                  Commits
                </Translate>
              </div>
            </Col>
          </Row>
        </div>

        <div ref={chooseRef} className={clsx(styles.choose, styles.fadeIn, chooseVisible && styles.visible)}>
          <Image className={styles.img} src="https://img.alicdn.com/imgextra/i3/O1CN01NwkUFs26qbxZhsz2J_!!6000000007713-55-tps-54-54.svg" preview={false}></Image>
          <div className={styles.wrap}>
            <div className={styles.left}>
              <div>
                <Translate>Why</Translate>
              </div>
              <div>
                <Translate>Choose ROLL</Translate>
              </div>
              <div className={styles.collapse}>
                <Collapse
                  ghost
                  defaultActiveKey={['1']}
                  expandIcon={({ isActive }) => {
                    return (<div className={isActive ? styles.isActive : styles.default}></div>)
                  }}
                  items={[
                    {
                      key: '1',
                      label: <div className={styles.label}><Translate>Distributed Architecture</Translate></div>,
                      children: <div className={styles.content}>
                        <Translate>Ray-based distributed architecture supporting mainstream engines like vLLM, SGLang, Megatron-Core, seamlessly scaling from single machine to large GPU clusters</Translate>
                      </div>,
                    },
                    {
                      key: '2',
                      label: <div className={styles.label}><Translate>Multi-task Learning</Translate></div>,
                      children: <div className={styles.content}>
                        <Translate>Support for multi-task joint training including math reasoning, code generation, and dialogue, with dynamic sampling rate and data weight adjustment</Translate>
                      </div>,
                    },
                    {
                      key: '3',
                      label: <div className={styles.label}><Translate>Extremely Easy to Use</Translate></div>,
                      children: <div className={styles.content}>
                        <Translate>Gym-style clean API design with modular architecture for flexible extension, one-click switching between different backend engines and algorithm configurations</Translate>
                      </div>,
                    },
                  ]}></Collapse>
              </div>
            </div>
            <div className={styles.right}>
              <Image className={styles.img} src="https://img.alicdn.com/imgextra/i1/O1CN01ATICVH28DLERvLxi3_!!6000000007898-2-tps-1388-1228.png" preview={false}></Image>
            </div>
          </div>
        </div>

        <div ref={coreRef} className={clsx(styles.core, styles.fadeIn, coreVisible && styles.visible)} id="core">
          <Image className={styles.img} src="https://img.alicdn.com/imgextra/i4/O1CN012UiWaS1KzDqk2EhyO_!!6000000001234-55-tps-54-54.svg" preview={false}></Image>
          <div>
            <div className={styles.title}>
              <Translate>
                Core Advantages
              </Translate>
            </div>
            <div className={styles.content}>
              <Translate>
                ROLL framework provides comprehensive reinforcement learning support, from model training to agent deployment, every aspect is carefully optimized to make AI training more efficient
              </Translate>
            </div>
          </div>
          <div className={styles.wrap}>
            <Divider style={{ borderColor: 'var(--home-divider-color)', marginBottom: 0 }}></Divider>
            <Row gutter={[0, 0]} align="bottom">
              <Col span={12}>
                <div className={styles.items}>
                  <div className={styles.label}>
                    <Translate>Born for Scale</Translate>
                  </div>
                  <div className={styles.content}>
                    <Translate>
                      Built on a Ray-based distributed architecture, it supports large-scale cluster training at the thousand-GPU level. Its innovative Rollout scheduler and AutoDeviceMapping module dramatically improve GPU resource utilization .
                    </Translate>
                  </div>
                </div>
              </Col>
              <Col span={12}>
                <div className={styles.items} style={{ paddingLeft: 30, borderRight: 'none' }}>
                  <div className={styles.label}>
                    <Translate>
                      Extreme Training Efficiency
                    </Translate>
                  </div>
                  <div className={styles.content}>
                    <Translate>
                      Integrates cutting-edge technologies like Megatron-Core, SGLang, and vLLM to significantly accelerate the model training and inference sampling processes .
                    </Translate>
                  </div>
                </div>
              </Col>
            </Row>
            <Row gutter={[0, 0]} align="bottom">
              <Col span={12}>
                <div className={styles.items} style={{ borderBottom: 'none' }}>
                  <div className={styles.label}>
                    <Translate>
                      Rich Algorithms & Scenarios
                    </Translate>
                  </div>
                  <div className={styles.content}>
                    <Translate>
                      Comes with built-in mainstream RL algorithms like PPO and GRPO, and supports multi-task RL and agent interaction scenarios. Its effectiveness has been validated in numerous real-world business applications .
                    </Translate>
                  </div>
                </div>
              </Col>
              <Col span={12}>
                <div className={styles.items} style={{ paddingLeft: 30, borderRight: 'none', borderBottom: 'none' }}>
                  <div className={styles.label}>
                    <Translate>
                      Open Source and Accessible
                    </Translate>
                  </div>
                  <div className={styles.content}>
                    <Translate>
                      ROLL is open-sourced on GitHub (https://github.com/alibaba/ROLL) under the Apache License 2.0, backed by an active community and comprehensive documentation .
                    </Translate>
                  </div>
                </div>
              </Col>
            </Row>
          </div>
        </div>

        <div ref={researchRef} className={clsx(styles.research, styles.fadeIn, researchVisible && styles.visible)} id="research">
          <Image className={styles.img} src="https://img.alicdn.com/imgextra/i1/O1CN016cZT1g1tk7L6GRZ7y_!!6000000005939-55-tps-54-54.svg" preview={false}></Image>
          <div>
            <div className={styles.title}>
              <Translate>
                Open Source Community
              </Translate>
            </div>
            <div className={styles.content}>
              <Translate>
                Join our vibrant open source community, explore cutting-edge reinforcement learning technologies with global AI researchers, and jointly promote the future of LLM and RL
              </Translate>
            </div>
          </div>
          <div className={styles.cards}>
            <div className={styles.card}>
              <div className={styles.label}>
                <Translate>
                  How to Contribute
                </Translate>
              </div>
              <div>
                <p>
                  <Translate>
                    Contribute algorithm implementations and performance optimizations
                  </Translate>
                </p>
                <p><Translate>Share experimental results and best practices</Translate></p>
                <p><Translate>Improve tutorials and learning resources</Translate></p>
              </div>
            </div>
            <div className={styles.card2} style={{ width: 300 }}>
              <div className={styles.label}>
                <Translate>
                  Join Discussion
                </Translate>
              </div>
              <div className={styles.buttons}>
                <Button className={styles.btn} onClick={() => setOpen(true)} icon={<WechatOutlined />}><Translate>WeChat</Translate></Button>
                <Button className={styles.github} target='_blank' href="https://github.com/alibaba/ROLL" variant="outlined" icon={<GithubOutlined />}>
                  <Translate>
                    Follow GitHub Repository
                  </Translate>
                </Button>
                <Button target='_blank' href="https://x.com/FutureLab2025" icon={<XOutlined />} className={styles.github} variant='outlined'>
                  <Translate>
                    Follow us on X
                  </Translate>
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
      <Modal
        open={open}
        onCancel={() => setOpen(false)}
        footer={null}
        getContainer={() => document.getElementById('home') || document.body}
      >
        <Image className={styles.whiteImg} src="https://img.alicdn.com/imgextra/i4/O1CN01MICK0T28fHMzy5P84_!!6000000007959-2-tps-756-850.png" preview={false} />
      </Modal>
    </div>
  </ConfigProvider>
}