import React from 'react';
import { Redirect } from '@docusaurus/router';
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";

export default function NotFoundContent() {
  const { i18n } = useDocusaurusContext();
  const { currentLocale } = i18n;
  const isChinese = currentLocale !== 'en';

  return <Redirect to={`/ROLL/${isChinese ? 'zh-Hans/' : ''}docs/Overview`} />;
}
