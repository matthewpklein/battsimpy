clean:
	find . -type f -name '*~' -delete
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*fuse*' -delete
	find . -type f -name '*.fuse*' -delete
	find . -type f -name '._*' -delete
