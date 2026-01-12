#!/usr/bin/env node
// scripts/update-stats.js
const fs = require('fs');

const OUTPUT_PATH = process.env.STATS_FILE || 'build/stats.json';

async function main() {
  const { Octokit } = await import('octokit');

  const token = process.env.GITHUB_TOKEN;
  const repo = process.env.REPO; // format: owner/repo

  if (!token || !repo) {
    throw new Error('Missing GITHUB_TOKEN or REPO environment variable');
  }

  const [owner, repoName] = repo.split('/');
  const octokit = new Octokit({ auth: token });

  console.log(`Fetching stats for ${owner}/${repoName}...`);

  // 1. Repo basic info
  const repoData = await octokit.rest.repos.get({ owner, repo: repoName });
  const stars = repoData.data.stargazers_count;
  const forks = repoData.data.forks_count;

  // 2. Contributors (all pages)
  const contributors = await octokit.paginate(octokit.rest.repos.listContributors, {
    owner,
    repo: repoName,
    per_page: 100,
  });
  const contributorCount = contributors.length;

  // 3. Issues (all & open) — includes PRs
  const issuesAll = await octokit.paginate(octokit.rest.issues.listForRepo, {
    owner,
    repo: repoName,
    state: 'all',
    per_page: 100,
  });
  const issuesOpen = await octokit.paginate(octokit.rest.issues.listForRepo, {
    owner,
    repo: repoName,
    state: 'open',
    per_page: 100,
  });

  const totalIssues = issuesAll.length;
  const openIssues = issuesOpen.length;

  // 4. Pull Requests (all & open)
  const prsAll = await octokit.paginate(octokit.rest.pulls.list, {
    owner,
    repo: repoName,
    state: 'all',
    per_page: 100,
  });
  const prsOpen = await octokit.paginate(octokit.rest.pulls.list, {
    owner,
    repo: repoName,
    state: 'open',
    per_page: 100,
  });

  const totalPRs = prsAll.length;
  const openPRs = prsOpen.length;

  // 5. Commits (all)
  const commits = await octokit.paginate(octokit.rest.repos.listCommits, {
    owner,
    repo: repoName,
    per_page: 100,
  });
  const commitCount = commits.length;

  // Pure issues = total issues - PRs
  const pureTotalIssues = totalIssues - totalPRs;
  const pureOpenIssues = openIssues - openPRs;

  const date = new Date().toISOString().split('T')[0]; // YYYY-MM-DD

  // Load existing data
  let stats = {};
  if (fs.existsSync(OUTPUT_PATH)) {
    try {
      stats = JSON.parse(fs.readFileSync(OUTPUT_PATH, 'utf8'));
    }
    catch (err) {
      console.log('fail to get stats file')
    }
  }

  stats[date] = {
    stars,
    forks,
    contributors: contributorCount,
    commits: commitCount,
    issues: {
      total: pureTotalIssues,
      open: pureOpenIssues,
      fixRate: parseInt(100 - (pureOpenIssues / pureTotalIssues) * 100, 10),
    },
    prs: {
      total: totalPRs,
      open: openPRs,
    },
  };

  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(stats, null, 2));
  console.log(`✅ Stats updated for ${date}`);
}

main().catch(err => {
  console.error('❌ Error:', err.message);
  process.exit(1);
});