Idea, what to do for each runner:

- create a sample job
- create a template job
- create a tool to quickly add an arbitrary python job

- [ ] local plist
  - [ ] write a tool to setup job easily
  - [ ] add a helpful alias / docstring to the tool 
  - [ ] Find a way to receive notifications
- [ ] local docker
- [ ] local script runner

# TODO

- [x] scheduler is supposed to be running continuously. Checking which jobs are due to run and running them (well, sleep 1 sec or 1 min to avoid overload. Also, make sure jobs don't block each other)
- [x] scheduler should have fastapi handles for adding new jobs @README.md
- [x] we need a convenient util that would go to scheduler on a specified port and put the job into scheduler via request