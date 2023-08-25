# Readme

Software for the Paper "Fully Tally-Hiding Verifiable E-Voting for Real-World Elections with Seat-Allocations"

Install all necessary requirements via

```bash
python3 -m pip install -r requirements.txt
```

Run the software by starting the servers:

```bash
python3 start_bb_server.py
```

```bash
python3 start_trustee_server.py
```

and then start the evaluation of the election:

```bash
python3 main.py
```
