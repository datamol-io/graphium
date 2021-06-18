# Configuration

$PROJECT = $GITHUB_REPO  = 'goli'
$GITHUB_ORG = 'valence-discovery'
$PUSH_TAG_REMOTE = 'git@github.com:valence-discovery/goli.git'

# Logic

$AUTHORS_FILENAME = 'AUTHORS.rst'
$AUTHORS_METADATA = '.authors.yml'
$AUTHORS_SORTBY = 'alpha'
$AUTHORS_MAILMAP = '.mailmap'

$CHANGELOG_FILENAME = 'CHANGELOG.rst'
$CHANGELOG_TEMPLATE = 'TEMPLATE.rst'
$CHANGELOG_NEWS = 'news'

$FORGE_FEEDSTOCK_ORG = 'valence-forge'
$FORGE_RERENDER = True
$FORGE_USE_GIT_URL = True
$FORGE_FORK = False
$FORGE_PULL_REQUEST = False

$ACTIVITIES = ['check', 'authors', 'changelog', 'version_bump', 'tag', 'push_tag', 'ghrelease']

$VERSION_BUMP_PATTERNS = [('goli/_version.py', r'__version__\s*=.*', "__version__ = \"$VERSION\""),
                          ('setup.py', r'version\s*=.*,', "version=\"$VERSION\",")
                          ]
