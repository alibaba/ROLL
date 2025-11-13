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
import { useHistory } from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

export default function Navbar() {
  const {
    navbar: { title, logo },
  } = useThemeConfig();
  const { colorMode, setColorMode } = useColorMode();
  const location = useLocation();
  const history = useHistory();
  const { i18n } = useDocusaurusContext();
  const { currentLocale } = i18n;
  const isChinese = currentLocale !== 'en';

  return (
    <ConfigProvider theme={{ algorithm: colorMode === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm }}>
      <nav className={clsx('navbar', 'navbar--fixed-top', styles.navbar)}>
        <div className={clsx(location.pathname === '/ROLL/' ? "container" : '', "navbar__inner")}>
          {/* 左侧 Logo 和标题 */}
          <div className={clsx(styles.logoWrap, 'navbar__items')} onClick={() => {
            window.location.href = '/ROLL/'
          }}>
            <div className={styles.logo}>
              <Image height={32} width={40} src={useBaseUrl(logo?.src)} alt="ROLL" preview={false} />
            </div>
            <div>
              <div className={styles.title}>
                {title}
              </div>
              <div className={styles.subTitle}>like a Reinforcement Learning Algorithm Developer</div>
            </div>
          </div>

          {/* 右侧导航项 */}
          <div className="navbar__items navbar__items--right">
            <Button className={clsx(styles.btn, location.pathname === '/ROLL/' ? styles.primary : '')} href="/ROLL/" type="text">Home</Button>
            <Button className={clsx(styles.btn, location.pathname === '/ROLL/' && location.hash === '#core' ? styles.primary : '')} href="/ROLL/#core" type="text">Core Algorithms</Button>
            <Button className={clsx(styles.btn, location.pathname === '/ROLL/' && location.hash === '#research' ? styles.primary : '')} href="/ROLL/#research" type="text">Research Community</Button>
            <Button className={clsx(styles.btn, location.pathname !== '/ROLL/' ? styles.primary : '')} type="text" href="/ROLL/docs/UserGuide/start">API Docs</Button>
            <Button className={styles.btn} href='https://github.com/alibaba/ROLL' type="text">Github<ExportOutlined /></Button>
            {
              location.pathname !== '/ROLL/' &&
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
            }
            {
              location.pathname !== '/ROLL/' &&
              <SearchBar />
            }
            {
              location.pathname === '/ROLL/' &&
              <Button className={styles.language} icon={<GlobalOutlined />}>English</Button>
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
