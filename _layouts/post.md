---
layout: default
---
<head>
  {% unless page.no_latex %}
    {% include latex.html %}
  {% endunless %}
</head>
<h1>{{ page.title }}</h1>
<p class="meta">{{ page.date | date_to_string }}</p>

<div class="post" markdown="1">
  {{ content }}
</div>
