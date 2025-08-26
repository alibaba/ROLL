import React from 'react';
import {useCurrentSidebarCategory} from '@docusaurus/theme-common';

function CategoryLink({ item, subLabel }) {
  if (item.type === 'link') {
    return <div><a href={item.href}>{item.label}</a></div>;
  }

  if (item.type === 'category') {
    return (<div>
      {
        item.label !== subLabel &&
        <h4 style={{ marginTop: 8 }}>{item.label}</h4>
      }
      <div>
        {item.items.sort((a) => a.type === 'link' ? -1 : 1).map(subItem => <CategoryLink item={subItem} subLabel={subLabel} />)}
      </div>
    </div>)
  }

  return <div>aa</div>
}

function LinksOfFolder({ folder_label }) {
  const category = useCurrentSidebarCategory();

  return (
    <div>
      {/* 这里不再显示这个标题，改为在start里展示，方便对应的页面展示出子导航 */}
      {/* <h3>{folder_label}</h3> */}
      {category.items.filter(item => item.label === folder_label).sort((a) => a.type === 'link' ? 1 : -1).map(item => <CategoryLink item={item} subLabel={folder_label} />)}
    </div>
  );
}



export default LinksOfFolder;