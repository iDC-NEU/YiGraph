import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Translate, { translate } from '@docusaurus/Translate';
import styles from './index.module.css';

export default function Home() {
  const { i18n } = useDocusaurusContext();
  const currentLocale = i18n.currentLocale;

  return (
    <Layout 
      title={translate({
        id: 'homepage.title',
        message: 'YiGraph',
        description: 'The homepage title'
      })}
      description={translate({
        id: 'homepage.description',
        message: 'YiGraph - 面向复杂关联数据的图分析智能体系统',
        description: 'The homepage description'
      })}
    >
      <main className={styles.hero}>
        <div className={styles.heroContent}>
          <h1 className={styles.heroTitle}>
            <Translate id="homepage.heroTitle">YiGraph</Translate>
          </h1>
          <h2 className={styles.heroSubtitle}>
            <Translate id="homepage.heroSubtitle">
              面向复杂关联数据的图分析智能体系统
            </Translate>
          </h2>
          <p className={styles.heroTagline}>
            <Translate id="homepage.heroTagline">
              用自然语言，洞察数据背后的深层关系
            </Translate>
          </p>
          
          <div className={styles.heroButtons}>
            <Link className={styles.primaryBtn} to="/docs/intro">
              <Translate id="homepage.introBtn">简介</Translate>
            </Link>
            <Link className={styles.secondaryBtn} to="/docs/intro">
              <Translate id="homepage.quickStartBtn">快速开始</Translate>
            </Link>
            <a className={styles.githubBtn} href="https://github.com/superccy/AAG" target="_blank" rel="noopener noreferrer">
              <Translate id="homepage.githubBtn">Github →</Translate>
            </a>
          </div>
        </div>
      </main>
    </Layout>
  );
}
