import React from 'react';
import Layout from '@theme-original/Layout';
import AIChat from '@site/src/components/AIChat';

export default function CustomLayout(props) {
  return (
    <>
      <Layout {...props} />
      <AIChat />
    </>
  );
}