<!DOCTYPE html>
<html>
	<head>
		<title>{{ page.title }}</title>
		<!-- link to main stylesheet -->
		<link rel="stylesheet" type="text/css" href="/css/main.css">
	</head>
	<body>
		<nav>
    		<ul>
        		<li><a href="/">Home</a></li>
	        	<li><a href="/about">About</a></li>
        		<li><a href="/cv/CV.pdf">CV</a></li>
        		<li><a href="/blog">Blog</a></li>
    		</ul>
		</nav>
		<div class="container">
    		{{ content }}
		</div>
		<footer>
    		<ul>
        		<li><a href="stevenwu3721@163.com">email</a></li>
        		<li><a href="https://github.com/Si1w">Github</a></li>
			</ul>
		</footer>
	</body>
</html>