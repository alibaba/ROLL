import React from 'react';
import { Button, ConfigProvider, theme, Image, Dropdown } from 'antd';
import { SunOutlined, MoonOutlined, GlobalOutlined, ExportOutlined } from '@ant-design/icons';
import clsx from 'clsx';
import { useThemeConfig } from '@docusaurus/theme-common';
import { useColorMode } from '@docusaurus/theme-common';
import styles from './styles.module.css';
import useBaseUrl from '@docusaurus/useBaseUrl';
import { useLocation } from '@docusaurus/router';
import SearchBar from '@theme/SearchBar'
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Translate from '@docusaurus/Translate';

export default function Navbar() {
  const {
    navbar: { title, logo },
  } = useThemeConfig();
  const { colorMode, setColorMode } = useColorMode();
  const location = useLocation();
  const { i18n } = useDocusaurusContext();
  const { currentLocale } = i18n;
  const isChinese = currentLocale !== 'en';
  const isHomePage = location.pathname === '/ROLL/' || location.pathname === '/ROLL/zh-Hans/';
  const targetPath = isChinese ? '/ROLL/zh-Hans/' : '/ROLL/'

  return (
    <ConfigProvider theme={{ algorithm: colorMode === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm }}>
      <nav className={clsx('navbar', 'navbar--fixed-top', styles.navbar)}>
        <div className={clsx(isHomePage ? "container" : '', "navbar__inner")}>
          {/* 左侧 Logo 和标题 */}
          <div className={clsx(styles.logoWrap, 'navbar__items')} onClick={() => {
            window.location.href = targetPath;
          }}>
            <div className={styles.logo}>
              <Image height={32} width={40} src={useBaseUrl(logo?.src)} alt="ROLL" preview={false} />
            </div>
            <div>
              <div className={styles.title}>
                {title}
              </div>
              <div className={styles.subTitle}>
                <Translate>like a Reinforcement Learning Algorithm Developer</Translate>
              </div>
            </div>
          </div>

          {/* 右侧导航项 */}
          <div className="navbar__items navbar__items--right">
            <Button className={clsx(styles.btn, isHomePage ? styles.primary : '')} href={targetPath} type="text">
              <Translate>Home</Translate>
            </Button>
            <Button className={clsx(styles.btn, isHomePage && location.hash === '#core' ? styles.primary : '')} href={`${targetPath}#core`} type="text">
              <Translate>Core Algorithms</Translate>
            </Button>
            <Button className={clsx(styles.btn, isHomePage && location.hash === '#research' ? styles.primary : '')} href={`${targetPath}#research`} type="text">
              <Translate>Research Community</Translate>
            </Button>
            <Button className={clsx(styles.btn, !isHomePage ? styles.primary : '')} type="text" href={`${targetPath}docs/UserGuide/start`}>
              <Translate>API Docs</Translate>
            </Button>
            <Button className={styles.btn} href='https://github.com/alibaba/ROLL' type="text">Github<ExportOutlined /></Button>
            <Dropdown
              menu={{
                items: [
                  {
                    key: 'en',
                    label: 'English',
                    disabled: !isChinese,
                    onClick: () => {
                      if (!isChinese) {
                        return;
                      }

                      window.location.href = location.pathname.replace('/zh-Hans/', '/');
                    }
                  },
                  {
                    key: 'zh-Hans',
                    label: '简体中文',
                    disabled: isChinese,
                    onClick: () => {
                      if (isChinese) {
                        return;
                      }

                      const paths = location.pathname.split('/ROLL/');
                      const newPath = `/ROLL/zh-Hans/${paths[1]}`;

                      window.location.href = newPath;
                    },
                  },
                ]
              }}>
              <Button className={styles.language} icon={<GlobalOutlined />}>{
                isChinese ? '简体中文' : 'English'
              }</Button>
            </Dropdown>
            {
              !isHomePage &&
              <SearchBar />
            }
            <Button
              onClick={() => setColorMode(colorMode === 'dark' ? 'light' : 'dark')}
              type="text"
              icon={colorMode === 'dark' ? <SunOutlined style={{ fontSize: '20px' }} /> : <MoonOutlined style={{ fontSize: '20px' }} />}
              style={{ marginLeft: 6 }}
            >
            </Button>
          </div>
        </div>
      </nav>
    </ConfigProvider>

  );
}
