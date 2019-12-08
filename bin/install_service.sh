#!/bin/bash

cp -f nemoware-analyzer.service /lib/systemd/system/
chown root:root /lib/systemd/system/nemoware-analyzer.service
systemctl daemon-reload
systemctl enable nemoware-analyzer.service
systemctl start nemoware-analyzer.service
