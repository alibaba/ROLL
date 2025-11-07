import React from 'react';

import styles from './styles.module.css';

export default ({ count, content }) => {
  return <div className={styles.container}>
    <div className={styles.count}>{count}</div>
    <div className={styles.content}>{content}</div>
  </div>
}