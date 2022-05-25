select distinct symbol from rates;
select
(select count(*) from rates where symbol='AUDUSD') union
(select count(*) from rates where symbol='GBPAUD') union
(select count(*) from rates where symbol='GBPNZD') union
(select count(*) from rates where symbol='EURGBP') union
(select count(*) from rates where symbol='AUDCHF') union
(select count(*) from rates where symbol='AUDNZD') union
(select count(*) from rates where symbol='NZDUSD')

