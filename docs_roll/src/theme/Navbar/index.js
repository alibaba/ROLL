import React from 'react';
import { Avatar, Button, ConfigProvider, theme, Image } from 'antd';
import { SunOutlined, MoonOutlined, GlobalOutlined, ExportOutlined } from '@ant-design/icons';
import clsx from 'clsx';
import { useThemeConfig } from '@docusaurus/theme-common';
import { useColorMode } from '@docusaurus/theme-common';
import styles from './styles.module.css';
import useBaseUrl from '@docusaurus/useBaseUrl';
import { useLocation } from '@docusaurus/router';
import SearchBar from '@theme/SearchBar'
import { useHistory } from '@docusaurus/router';

export default function Navbar() {
  const {
    navbar: { title, logo },
  } = useThemeConfig();
  const { colorMode, setColorMode } = useColorMode();
  const location = useLocation();
  const history = useHistory();

  return (
    <ConfigProvider theme={{ algorithm: colorMode === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm }}>
      <nav className={clsx('navbar', 'navbar--fixed-top', styles.navbar)}>
        <div className={clsx(location.pathname === '/ROLL/' ? "container" : '', "navbar__inner")}>
          {/* 左侧 Logo 和标题 */}
          <div className={clsx(styles.logoWrap, 'navbar__items')} onClick={() => history.push('/ROLL/')}>
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
            <Button className={clsx(styles.btn, location.pathname !== '/ROLL/' ? styles.primary : '')} type="text" href="/ROLL/docs/English/start">API Docs</Button>
            <Button className={styles.btn} href='https://github.com/alibaba/ROLL' type="text">Github<ExportOutlined /></Button>
            {
              location.pathname !== '/ROLL/' &&
              <SearchBar />
            }
            {
              location.pathname === '/ROLL/' &&
              <Button className={styles.language} icon={<GlobalOutlined />}>英文</Button>
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
