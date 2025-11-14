import React, { useState } from 'react';
import { Button, Image, Divider, Col, Row, Collapse, Modal, ConfigProvider, theme } from 'antd';
import { GithubOutlined, WechatOutlined } from '@ant-design/icons';
import clsx from 'clsx';
import useBaseUrl from '@docusaurus/useBaseUrl';
import { useColorMode } from '@docusaurus/theme-common';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Translate, { translate } from '@docusaurus/Translate';
import Statistic from '../Statistic';

import styles from './styles.module.css';

export default () => {
  const [open, setOpen] = useState(false);
  const { colorMode } = useColorMode();
  const { i18n } = useDocusaurusContext();
  const { currentLocale } = i18n;
  const isChinese = currentLocale !== 'en';
  const targetPath = isChinese ? '/ROLL/zh-Hans/' : '/ROLL/'

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
          <Button href={`${targetPath}docs/UserGuide/start/`} target='_blank' className={styles.btn}>
            <Translate>{"Get Started >"}</Translate>
          </Button>
          <Button className={styles.github} target='_blank' href="https://github.com/alibaba/ROLL" variant="outlined" icon={<GithubOutlined />}>{"Github >"}</Button>
          <Button className={styles.github} target='_blank' href="https://deepwiki.com/alibaba/ROLL" variant="outlined" icon={<Image width={14} src={useBaseUrl('/img/deepwiki.svg')} preview={false}></Image>}>{"DeepWiki >"}</Button>
        </div>

        <div className={styles.mainImg}>
          <Image className={styles.img} src={useBaseUrl('/img/home_main.png')} preview={false}></Image>
        </div>

        <div className={styles.overview}>
          <div className={styles.left}>
            <Image className={styles.img} src={useBaseUrl('/img/icon1.svg')} preview={false}></Image>
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

        <div>
          <Row gutter={16}>
            <Col span={8}>
              <Statistic count="1.9k" content={translate({
                message: 'Github Stars',
              })} />
            </Col>
            <Col span={8}>
              <Statistic count="30+" content={translate({
                message: 'Contributors',
              })} />
            </Col>
            <Col span={8}>
              <Statistic count="200+" content={translate({
                message: 'Commits',
              })} />
            </Col>
          </Row>
        </div>

        <div className={styles.choose}>
          <Image className={styles.img} src={useBaseUrl('/img/icon2.svg')} preview={false}></Image>
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
              <Image className={styles.img} src={useBaseUrl('/img/choose.png')} preview={false}></Image>
            </div>
          </div>
        </div>

        <div className={styles.core} id="core">
          <Image className={styles.img} src={useBaseUrl('/img/icon3.svg')} preview={false}></Image>
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

        <div className={styles.research} id="research">
          <Image className={styles.img} src={useBaseUrl('/img/icon4.svg')} preview={false}></Image>
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
                <Button className={styles.github} href="https://github.com/alibaba/ROLL" variant="outlined" icon={<GithubOutlined />}>
                  <Translate>
                    Follow GitHub Repository
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