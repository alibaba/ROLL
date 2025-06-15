import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'For Tech Pioneers',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
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
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
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
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
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

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        {
          Svg && <Svg className={styles.featureSvg} role="img" />
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
