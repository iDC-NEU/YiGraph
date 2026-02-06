import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './index.module.css';

export default function Home() {
  return (
    <Layout title="AAG" description="AAG - 以数据为中心的AI系统">
      <main className={styles.hero}>
        <div className={styles.heroContent}>
          <h1 className={styles.heroTitle}>AAG</h1>
          <h2 className={styles.heroSubtitle}>以数据为中心的AI系统</h2>
          <p className={styles.heroTagline}>好数据，好模型</p>
          
          <div className={styles.heroButtons}>
            <Link className={styles.primaryBtn} to="/docs/intro">
              简介
            </Link>
            <Link className={styles.secondaryBtn} to="/docs/intro">
              快速开始
            </Link>
            <a className={styles.githubBtn} href="https://github.com/superccy/AAG" target="_blank" rel="noopener noreferrer">
              Github →
            </a>
          </div>
        </div>
      </main>
    </Layout>
  );
}
