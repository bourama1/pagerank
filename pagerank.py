import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print("PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    n = len(corpus)

    # current page has no outgoing links = link to every page
    links = corpus[page] if corpus[page] else set(corpus.keys())

    # base probability from the random-teleport
    base = (1 - damping_factor) / n

    # Pages linked from current page
    link_bonus = damping_factor / len(links)

    return {p: base + (link_bonus if p in links else 0) for p in corpus}


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    counts = {page: 0 for page in corpus}

    # First sample: pick a page completely at random
    page = random.choice(list(corpus.keys()))

    for _ in range(n):
        counts[page] += 1

        # Get the probability distribution over next pages given current page
        dist = transition_model(corpus, page, damping_factor)

        # Randomly pick the next page weighted by the transition probabilities.
        page = random.choices(list(dist.keys()), weights=dist.values())[0]

    return {page: count / n for page, count in counts.items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    n = len(corpus)

    # Start with a uniform distribution: every page is equally likely
    ranks = {page: 1 / n for page in corpus}

    # Pre-process: pages with no outgoing links = link to every page
    adjusted = {
        page: (links if links else set(corpus.keys())) for page, links in corpus.items()
    }

    while True:
        new_ranks = {}

        for page in corpus:
            # Sum contributions from every page that links to this page.
            link_sum = sum(
                ranks[p] / len(adjusted[p])
                for p in corpus
                if page in adjusted[p]  # only pages that actually link here
            )

            # PR(page) = (1 - d) / N  +  d * sum(PR(i) / NumLinks(i))
            new_ranks[page] = (1 - damping_factor) / n + damping_factor * link_sum

        # Check convergence
        if all(abs(new_ranks[p] - ranks[p]) < 0.001 for p in corpus):
            return new_ranks

        ranks = new_ranks


if __name__ == "__main__":
    main()
