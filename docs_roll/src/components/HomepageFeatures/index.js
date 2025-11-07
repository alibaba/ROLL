import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'For Tech Pioneers',
    img: 'img/pioneer.png',
    description: (
      <>
        Fast and Cost-Effective.
        <br />
        Scalability and Fault Tolerance.
        <br />
        Flexible Hardware Usage.
      </>
    ),
  },
  {
    title: 'For Product Developers',
    // Svg: require('@site/static/img/develop.svg').default,
    img: 'img/develop.png',
    description: (
      <>
        Diverse and Extensible Rewards/Environments.
        <br />
        Compositional Sample-Reward Route.
        <br />
        Easy Device-Reward Mapping.
      </>
    ),
  },
  {
    title: 'For Algorithm Researchers',
    // Svg: require('@site/static/img/researcher.svg').default,
    img: 'img/researcher.png',
    description: (
      <>
        Constrained Device Execution.
        <br />
        Pluggable RLVR & Agentic RL Pipeline.
        <br />
        Transparent Experimentation
      </>
    ),
  },
];

function Feature({ Svg = '', title, description, img }) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        {
          Svg && <Svg className={styles.featureSvg} role="img" />
        }
        {
          img && <img src={img} alt={title} className={styles.featureSvg} />
        }
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
